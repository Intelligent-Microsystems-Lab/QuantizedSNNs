import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
import time
import math
import numpy as np
import pandas as pd
import argparse
import datetime

import quantization
from localQ import sparse_data_generator_DVS, smoothstep, superspike#, QLinearLayerSign, LIFDenseLayer, LIFConv2dLayer



class NQLinearFunctional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, scale):
        w_quant = weights
        bias_quant = bias

        output = torch.einsum("ab,bc->ac", (input, w_quant)) + bias_quant
        
        ctx.save_for_backward(input, w_quant, bias_quant)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w_quant, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        quant_error = grad_output

        if ctx.needs_input_grad[0]:
            grad_input = torch.einsum("ab,cb->ac", (quant_error, w_quant))
        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum("ab,ac->bc", (quant_error, input))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = quant_error.sum(0).squeeze(0)

        return grad_input, grad_weight.T, grad_bias, None

class NQLIFDenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, tau_syn, tau_mem, tau_ref, delta_t, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float):
        super(NQLIFDenseLayer, self).__init__()    
        self.device = device  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.thr = thr
        self.fan_in = in_channels
        self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(64)]))
        self.L = np.max([np.sqrt( 6/self.fan_in), self.L_min])
        self.scale = 2 ** round(math.log(self.L_min / self.L, 2.0))
        self.scale = self.scale if self.scale > 1 else 1.0

        if tau_syn.shape[0] == 2:
            self.beta = torch.exp( -delta_t / torch.Tensor(self.in_channels).uniform_(tau_syn[0], tau_syn[0]).to(device))
        else:
            self.beta = torch.Tensor([torch.exp( - delta_t / tau_syn)]).to(device)
        if tau_mem.shape[0] == 2:
            self.alpha = torch.exp( -delta_t / torch.Tensor(self.in_channels).uniform_(tau_mem[0], tau_mem[0]).to(device))
        else:
            self.alpha = torch.Tensor([torch.exp( - delta_t / tau_mem)]).to(device)

        if tau_ref.shape[0] == 2:
            self.gamma = torch.exp( -delta_t / torch.Tensor(self.out_channels).uniform_(tau_ref[0], tau_ref[0]).to(device))
        else:
            self.gamma = torch.Tensor([torch.exp( - delta_t / tau_ref)]).to(device)


        self.weights = nn.Parameter(torch.empty((self.in_channels, self.out_channels),  device=device, dtype=dtype, requires_grad=True))
        torch.nn.init.uniform_(self.weights, a = -self.L, b = self.L)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels,  device=device, dtype=dtype, requires_grad=True))
            torch.nn.init.uniform_(self.bias, a = -self.L, b = self.L)
        else:
            self.register_parameter('bias', None)


    def state_init(self, batch_size):
        self.P = torch.zeros(batch_size, self.in_channels).detach().to(self.device)
        self.Q = torch.zeros(batch_size, self.in_channels).detach().to(self.device)
        self.R = torch.zeros(batch_size, self.out_channels).detach().to(self.device)
        self.S = torch.zeros(batch_size, self.out_channels).detach().to(self.device)
        self.U = torch.zeros(batch_size, self.out_channels).detach().to(self.device)

    
    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t

        self.U = NQLinearFunctional.apply(self.P, self.weights, self.bias, self.scale) + self.R
        self.S = (self.U > self.thr).float()
        return self.S


class NQConv2dFunctional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, scale, padding = 0, pooling = None):
        w_quant = weights
        bias_quant = bias
        ctx.padding = padding
        ctx.pooling = pooling 
        ctx.size_pool = None
        pool_indices = torch.ones(0)

        output = F.conv2d(input = input, weight = w_quant, bias = bias_quant, padding = ctx.padding)
        
        if ctx.pooling is not None:
            mpool = nn.MaxPool2d(kernel_size = ctx.pooling, stride = ctx.pooling, padding = (ctx.pooling-1)//2, return_indices=True)
            ctx.size_pool = output.shape
            output, pool_indices = mpool(output)

        ctx.save_for_backward(input, w_quant, bias_quant, pool_indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w_quant, bias_quant, pool_indices = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None 
        unmpool = nn.MaxUnpool2d(ctx.pooling, stride = ctx.pooling, padding = (ctx.pooling-1)//2)

        if ctx.pooling is not None:
            grad_output = unmpool(grad_output, pool_indices, output_size = torch.Size(ctx.size_pool))
        quant_error = grad_output

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, w_quant, quant_error, padding = ctx.padding)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, w_quant.shape, quant_error, padding = ctx.padding)
        if bias_quant is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.einsum("abcd->b",(quant_error))

        return grad_input, grad_weight, grad_bias, None, None, None


class NQLIFConv2dLayer(nn.Module):
    def __init__(self, inp_shape, kernel_size, out_channels, tau_syn, tau_mem, tau_ref, delta_t, pooling = 1, padding = 0, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float):
        super(NQLIFConv2dLayer, self).__init__()   
        self.device = device
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.out_channels = out_channels  
        self.fan_in = kernel_size * kernel_size * inp_shape[0]
        self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(64)]))
        self.L = np.max([np.sqrt( 6/self.fan_in), self.L_min])
        self.scale = 2 ** round(math.log(self.L_min / self.L, 2.0))
        self.scale = self.scale if self.scale > 1 else 1.0

        self.padding = padding
        self.pooling = pooling
                
        self.weights = nn.Parameter(torch.empty((self.out_channels, inp_shape[0],  self.kernel_size, self.kernel_size),  device=device, dtype=dtype, requires_grad=True))
        torch.nn.init.uniform_(self.weights, a = -self.L, b = self.L)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype, requires_grad=True))
            torch.nn.init.uniform_(self.bias, a = -self.L, b = self.L)
        else:
            self.register_parameter('bias', None)

        self.out_shape = NQConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape).to(device), self.weights, self.bias, self.scale, self.padding, self.pooling).shape[1:]
        self.thr = thr

        if tau_syn.shape[0] == 2:
            self.beta = torch.exp( -delta_t / torch.Tensor(self.inp_shape).uniform_(tau_syn[0], tau_syn[0]).to(device))
        else:
            self.beta = torch.Tensor([torch.exp( - delta_t / tau_syn)]).to(device)
        if tau_mem.shape[0] == 2:
            self.alpha = torch.exp( -delta_t / torch.Tensor(self.inp_shape).uniform_(tau_mem[0], tau_mem[0]).to(device))
        else:
            self.alpha = torch.Tensor([torch.exp( - delta_t / tau_mem)]).to(device)

        if tau_ref.shape[0] == 2:
            self.gamma = torch.exp( -delta_t / torch.Tensor(self.out_shape).uniform_(tau_ref[0], tau_ref[0]).to(device))
        else:
            self.gamma = torch.Tensor([torch.exp( - delta_t / tau_ref)]).to(device)


    def state_init(self, batch_size):
        self.P = torch.zeros((batch_size,) + self.inp_shape).detach().to(self.device)
        self.Q = torch.zeros((batch_size,) + self.inp_shape).detach().to(self.device)
        self.R = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)
        self.S = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)
        self.U = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)

    
    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t

        self.U = NQConv2dFunctional.apply(self.P, self.weights, self.bias, self.scale, self.padding, self.pooling) + self.R
        self.S = (self.U > self.thr).float()

        return self.S


class LinearFAFunction(torch.autograd.Function):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_fa)
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias

class FALinear(nn.Module):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    def __init__(self, input_features, output_features, bias=True):
        super(FALinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # weight and bias for forward pass
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # fixed random weight and bias for FA backward pass does not need gradient
        self.weight_fa = torch.nn.Parameter(torch.FloatTensor(output_features, input_features), requires_grad=False)
        self.reset_lc_parameters(self)

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)
    
    @staticmethod
    def reset_lc_parameters(layer):
        #Random initialization with 50% variation is done here
        if hasattr(layer, 'weight_fa'):
            layer.weight_fa.data.normal_(1, .5)
            layer.weight_fa.data[layer.weight_fa.data<0] = 0
            layer.weight_fa.data[:] *= layer.weight.data[:]
        stdv = 1. / np.sqrt(layer.weight.size(1))
        layer.weight.data.uniform_(-stdv, stdv)
        if layer.bias is not None:
            layer.bias.data.uniform_(-stdv, stdv)



verbose_output = False
ap = argparse.ArgumentParser()
ap.add_argument("-dir", "--dir", type = str, help = "output dir")
args = vars(ap.parse_args())

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float

# load data
with open('data/train_dvs_gesture.pickle', 'rb') as f:
    data = pickle.load(f)
x_train = data[0]
y_train = data[1]

with open('data/test_dvs_gesture.pickle', 'rb') as f:
    data = pickle.load(f)
x_test = data[0]
y_test = data[1]

# set quant level
quantization.global_wb = 64
quantization.global_ub = 64
quantization.global_qb = 64
quantization.global_pb = 64
quantization.global_gb = 64
quantization.global_eb = 64
quantization.global_rb = 16
quantization.global_lr = 1
quantization.global_beta = 1.5 #quantization.step_d(quantization.global_wb)-.5

# set parameters
ms = 1e-3
delta_t = 1*ms

T = 500*ms
T_test = 1800*ms
burnin = 50*ms
batch_size = 72
output_neurons = 11

tau_mem = torch.Tensor([20*ms]).to(device)#torch.Tensor([5*ms, 35*ms]).to(device)
tau_syn = torch.Tensor([7.5*ms]).to(device)#torch.Tensor([5*ms, 10*ms]).to(device)
tau_ref = torch.Tensor([0*ms]).to(device)
thr = torch.Tensor([.4]).to(device)

lambda1 = .2 
lambda2 = .1


# construct layers
dropout_p = .99
dropout_learning = nn.Dropout(p=dropout_p)

downsample = nn.AvgPool2d(kernel_size = 4, stride = 4)

layer1 = NQLIFConv2dLayer(inp_shape = (2, 32, 32), kernel_size = 5, out_channels = 16, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout1 = FALinear(np.prod(layer1.out_shape), output_neurons).to(device)

layer2 = NQLIFConv2dLayer(inp_shape = layer1.out_shape, kernel_size = 5, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout2 = FALinear(np.prod(layer2.out_shape), output_neurons).to(device)

layer3 = NQLIFConv2dLayer(inp_shape = layer2.out_shape, kernel_size = 5, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout3 = FALinear(np.prod(layer3.out_shape), output_neurons).to(device)

layer4 = NQLIFDenseLayer(in_channels = np.prod(layer3.out_shape), out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)

#log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
#nll_loss = torch.nn.NLLLoss()
sl1_loss = torch.nn.SmoothL1Loss()

all_parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters()) + list(layer4.parameters())

# initlialize optimizier
opt = torch.optim.SGD(all_parameters, lr=1e-9)

train_acc = []
test_acc = []


print("WPQUEG Quantization: {0}{1}{2}{3}{4}{5}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb))


for e in range(50):
    correct = 0
    total = 0
    tcorrect = 0
    ttotal = 0

    correct1_train = 0 # note over all time steps now...
    correct2_train = 0
    correct3_train = 0
    correct4_train = 0
    total_train = 0
    correct1_test = 0
    correct2_test = 0
    correct3_test = 0
    correct4_test = 0
    total_test = 0
    loss_hist = []

    start_time = time.time()

    rec_video = True
    for x_local, y_local in sparse_data_generator_DVS(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, shuffle = True, device = device):
        y_local = y_local -1
        y_onehot = torch.Tensor(len(y_local), output_neurons).to(device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)

        class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)

        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])
        layer4.state_init(x_local.shape[0])

        # burnin
        for t in range(int(burnin/ms)):
            down_spikes = downsample(x_local[:,:,:,:,t])*16

            # two channel trick
            down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
            #down_spikes = torch.cat((x_local[:,:,:,:,t], x_local[:,:,:,:,t]), dim = 1)
            mask1 = (down_spikes > 0) # this might change
            mask2 = (down_spikes < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            down_spikes = torch.zeros_like(down_spikes)
            down_spikes[mask1] = 1 
            down_spikes[mask2] = 1

            # same same but different
            # down_spikes[down_spikes != 0] = 1

            # bi directional


            # DTN

            out_spikes1 = layer1.forward(down_spikes)
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)

        # training
        for t in range(int(burnin/ms), int(T/ms)):
            total_train += y_local.size(0)
            loss_gen = 0
            down_spikes = downsample(x_local[:,:,:,:,t])*16

            # two channel trick
            down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
            #down_spikes = torch.cat((x_local[:,:,:,:,t], x_local[:,:,:,:,t]), dim = 1)
            mask1 = (down_spikes > 0) # this might change
            mask2 = (down_spikes < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            down_spikes = torch.zeros_like(down_spikes)
            down_spikes[mask1] = 1 
            down_spikes[mask2] = 1

            out_spikes1 = layer1.forward(down_spikes)
            rreadout1 = random_readout1(dropout_learning(smoothstep(layer1.U.reshape([x_local.shape[0], np.prod(layer1.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct1_train += (predicted == y_local).sum().item()
            loss_gen += sl1_loss(((rreadout1 / rreadout1.abs().max())+1)*.5, y_onehot) #+ lambda1 * F.relu(layer1.U+.01).mean() + lambda2 * F.relu(thr-layer1.U).mean()

            out_spikes2 = layer2.forward(out_spikes1)
            rreadout2 = random_readout2(dropout_learning(smoothstep(layer2.U.reshape([x_local.shape[0], np.prod(layer2.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout2.data, 1)
            correct2_train += (predicted == y_local).sum().item()
            loss_gen += sl1_loss( ((rreadout2 / rreadout2.abs().max())+1)*.5, y_onehot) #+ lambda1 * F.relu(layer2.U+.01).mean() + lambda2 * F.relu(thr-layer2.U).mean()

            out_spikes3 = layer3.forward(out_spikes2)
            rreadout3 = random_readout3(dropout_learning(smoothstep(layer3.U.reshape([x_local.shape[0], np.prod(layer3.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct3_train += (predicted == y_local).sum().item()
            loss_gen += sl1_loss( ((rreadout3 / rreadout3.abs().max())+1)*.5, y_onehot) #+ lambda1 * F.relu(layer3.U+.01).mean() + lambda2 * F.relu(thr-layer3.U).mean()

            # flattening for spiking readout layer
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)
            loss_gen += sl1_loss(smoothstep(layer4.U), y_onehot)
            _, predicted = torch.max(out_spikes4, 1)
            correct4_train += (predicted == y_local).sum().item()
            #y_log_p4 = log_softmax_fn(smoothstep(layer4.U))
            #gen_loss +=  sl1_loss(smoothstep(layer4.U), y_onehot) + lambda1 * F.relu(layer4.U+.01).mean() + lambda2 * F.relu(.1-layer4.U).mean()

            loss_gen.backward()
            opt.step()
            opt.zero_grad()

            loss_hist.append(loss_gen.item())
            class_rec += out_spikes4

        correct += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        total += len(y_local)
    train_time = time.time()


    # test accuracy
    for x_local, y_local in sparse_data_generator_DVS(x_test, y_test, batch_size = batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True):
        y_local = y_local -1
        class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)
        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])
        layer4.state_init(x_local.shape[0])

        # burnin
        for t in range(int(burnin/ms)):
            down_spikes = downsample(x_local[:,:,:,:,t])*16

            # two channel trick
            down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
            #down_spikes = torch.cat((x_local[:,:,:,:,t], x_local[:,:,:,:,t]), dim = 1)
            mask1 = (down_spikes > 0) # this might change
            mask2 = (down_spikes < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            down_spikes = torch.zeros_like(down_spikes)
            down_spikes[mask1] = 1 
            down_spikes[mask2] = 1

            out_spikes1 = layer1.forward(down_spikes)
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)

        # testing
        for t in range(int(burnin/ms), int(T_test/ms)):
            total_test += y_local.size(0)
            down_spikes = downsample(x_local[:,:,:,:,t])*16

            # two channel trick
            down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
            #down_spikes = torch.cat((x_local[:,:,:,:,t], x_local[:,:,:,:,t]), dim = 1)
            mask1 = (down_spikes > 0) # this might change
            mask2 = (down_spikes < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            down_spikes = torch.zeros_like(down_spikes)
            down_spikes[mask1] = 1 
            down_spikes[mask2] = 1

            # dropout kept active -> decolle note
            out_spikes1 = layer1.forward(down_spikes)
            rreadout1 = random_readout1(dropout_learning(smoothstep(layer1.U.reshape([x_local.shape[0], np.prod(layer1.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct1_test += (predicted == y_local).sum().item()

            out_spikes2 = dropout_learning(layer2.forward(out_spikes1)) 
            rreadout2 = random_readout2(dropout_learning(smoothstep(layer2.U.reshape([x_local.shape[0], np.prod(layer2.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout2.data, 1)
            correct2_test += (predicted == y_local).sum().item()

            out_spikes3 = dropout_learning(layer3.forward(out_spikes2))
            rreadout3 = random_readout3(dropout_learning(smoothstep(layer3.U.reshape([x_local.shape[0], np.prod(layer3.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct3_test += (predicted == y_local).sum().item()

            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)
            _, predicted = torch.max(out_spikes4, 1)
            correct4_test += (predicted == y_local).sum().item()

            class_rec += out_spikes4
        tcorrect += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        ttotal += len(y_local)
    inf_time = time.time()


    correct = correct.item()
    tcorrect = tcorrect.item()
    train_acc.append(correct/total)
    test_acc.append(tcorrect/ttotal)
    #print("Epoch {0} | Loss: {1:.4f} Train Acc: {2:.4f} Test Acc: {3:.4f} Train Time: {4:.4f}s Inference Time: {5:.4f}s".format(e+1, np.mean(loss_hist), correct.item()/total, tcorrect.item()/ttotal, train_time-start_time, inf_time - train_time)) 
    #np.mean(loss_hist2), np.mean(loss_hist3), np.mean(loss_hist4),
    if verbose_output:
        print("Epoch {0} | Loss: {1:.4f}, {2:.0f}, {3:.0f}, {4:.0f} Train Acc 1: {5:.4f} Test Acc 1: {6:.4f} Train Acc 2: {7:.4f} Test Acc 2: {8:.4f} Train Acc 3: {9:.4f} Test Acc 3: {10:.4f} Train Acc 4: {11:.4f} Test Acc 4: {12:.4f}  TRAIN_ACC: {13:.4f} TEST_ACC: {14:.4f}  Train Time: {15:.4f}s Inference Time: {16:.4f}s".format(e+1, np.mean(loss_hist), -1, -1, -1, correct1_train/total_train, correct1_test/total_train, correct2_train/total_train, correct2_test/total_train, correct3_train/total_train, correct3_test/total_train, correct4_train/total_train, correct4_test/total_train, correct/total, tcorrect/ttotal, train_time-start_time, inf_time - train_time))
    else:
        print("Epoch {0} | TRAIN_ACC: {1:.4f} TEST_ACC: {2:.4f}  Train Time: {3:.4f}s Inference Time: {4:.4f}s".format(e+1, correct/total, tcorrect/ttotal, train_time-start_time, inf_time - train_time))


# saving results/weights
results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer3':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer4':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'loss':[loss_hist]} # 'test_acc': test_acc, 'train_acc': train_acc, , 'train_idx':shuffle_idx_ta, 'test_idx':shuffle_idx_te
with open('hello.pkl', 'wb') as f:
    pickle.dump(results, f)

# Epoch 41 | Loss: 2.6689 Train Acc: 0.0816 Test Acc: 0.0833 Train Time: 734.5396s Inference Time: 298.9132s