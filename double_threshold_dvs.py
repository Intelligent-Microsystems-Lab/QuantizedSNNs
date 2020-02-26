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

import quantization
from localQ import sparse_data_generator_DVS, QLinearLayerSign, QSConv2dFunctional, QSLinearFunctional


class DTNSmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):     
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input[x <= -.5] = 0
        grad_input[x > .5] = 0

        # quantize error
        grad_input = quantization.quant_err(grad_input)

        return grad_input

DTNsmoothstep = DTNSmoothStep().apply


class DTNSuperSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, x):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input = grad_input/(SuperSpike.scale*torch.abs(x)+1.0)**2

        # quantize error
        grad_input = quantization.quant_err(grad_input)

        return grad_input

DTNsuperspike = DTNSuperSpike().apply


class DTNLIFDenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, tau_syn, tau_mem, tau_ref, delta_t, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float):
        super(DTNLIFDenseLayer, self).__init__()    
        self.device = device  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.thr = thr
        self.fan_in = in_channels
        self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_wb)]))
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

        with torch.no_grad():
            self.weights.data = quantization.clip(quantization.quant_generic(self.weights.data, quantization.global_gb)[0], quantization.global_wb)
            if self.bias is not None:
                self.bias.data = quantization.clip(quantization.quant_generic(self.bias.data, quantization.global_gb)[0], quantization.global_wb)

    def state_init(self, batch_size):
        self.P = torch.zeros(batch_size, self.in_channels).detach().to(self.device)
        self.Q = torch.zeros(batch_size, self.in_channels).detach().to(self.device)
        self.R = torch.zeros(batch_size, self.out_channels).detach().to(self.device)
        self.S = torch.zeros(batch_size, self.out_channels).detach().to(self.device)
        self.U = torch.zeros(batch_size, self.out_channels).detach().to(self.device)

    
    def forward(self, input_t):
        with torch.no_grad():
            self.weights.data = quantization.clip(self.weights.data, quantization.global_gb)
            if self.bias is not None:
                self.bias.data = quantization.clip(self.bias.data, quantization.global_gb)

        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t

        # quantize P, 
        self.P, _ = quantization.quant_generic(self.P, quantization.global_pb)
        self.Q, _ = quantization.quant_generic(self.Q, quantization.global_qb)
        

        self.U = QSLinearFunctional.apply(self.P, self.weights, self.bias, self.scale) + self.R
        self.S = (self.U > self.thr).float()

        # quantize U
        self.U, _ = quantization.quant_generic(self.U, quantization.global_ub)

        return self.S

class DTNLIFConv2dLayer(nn.Module):
    def __init__(self, inp_shape, kernel_size, out_channels, tau_syn, tau_mem, tau_ref, delta_t, pooling = 1, padding = 0, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float):
        super(DTNLIFConv2dLayer, self).__init__()   
        self.device = device
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.out_channels = out_channels  
        self.fan_in = kernel_size * kernel_size * inp_shape[0]
        self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_wb)]))
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

        self.out_shape = QSConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape).to(device), self.weights, self.bias, self.scale, self.padding, self.pooling).shape[1:]
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

        with torch.no_grad():
            self.weights.data = quantization.clip(quantization.quant_generic(self.weights.data, quantization.global_gb)[0], quantization.global_wb)
            if self.bias is not None:
                self.bias.data = quantization.clip(quantization.quant_generic(self.bias.data, quantization.global_gb)[0], quantization.global_wb)

    def state_init(self, batch_size):
        self.P = torch.zeros((batch_size,) + self.inp_shape).detach().to(self.device)
        self.Q = torch.zeros((batch_size,) + self.inp_shape).detach().to(self.device)
        self.R = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)
        self.S = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)
        self.U = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)

    
    def forward(self, input_t):
        with torch.no_grad():
            self.weights.data = quantization.clip(self.weights.data, quantization.global_gb)
            if self.bias is not None:
                self.bias.data = quantization.clip(self.bias.data, quantization.global_gb)

        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t

        # quantize P, Q
        self.P, _ = quantization.quant_generic(self.P, quantization.global_pb)
        self.Q, _ = quantization.quant_generic(self.Q, quantization.global_qb)

        self.U = QSConv2dFunctional.apply(self.P, self.weights, self.bias, self.scale, self.padding, self.pooling) + self.R
        self.S = (self.U > self.thr).float()

        # quantize U
        self.U, _ = quantization.quant_generic(self.U, quantization.global_ub)

        return self.S




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
with open('data/small_train_dvs_gesture.pickle', 'rb') as f:
    data = pickle.load(f)
x_train = data[0]
y_train = data[1]

with open('data/small_test_dvs_gesture.pickle', 'rb') as f:
    data = pickle.load(f)
x_test = data[0]
y_test = data[1]

# set quant level
quantization.global_wb = 8
quantization.global_ub = 8
quantization.global_qb = 8
quantization.global_pb = 8
quantization.global_gb = 8
quantization.global_eb = 8
quantization.global_rb = 16
quantization.global_lr = 1
quantization.global_beta = 1.5 #quantization.step_d(quantization.global_wb)-.5

# set parameters
ms = 1e-3
delta_t = 1*ms

T = 500*ms
T_test = 1800*ms
burnin = 50*ms
batch_size = 2 # 72
output_neurons = 11

tau_mem = torch.Tensor([20*ms]).to(device)#torch.Tensor([5*ms, 35*ms]).to(device)
tau_syn = torch.Tensor([7.5*ms]).to(device)#torch.Tensor([5*ms, 10*ms]).to(device)
tau_ref = torch.Tensor([0*ms]).to(device)
thr = torch.Tensor([.4]).to(device)

lambda1 = .2 
lambda2 = .1


# construct layers
dropout_learning = nn.Dropout(p=.5)

downsample = nn.AvgPool2d(kernel_size = 4, stride = 4)

layer1 = DTNLIFConv2dLayer(inp_shape = (2, 32, 32), kernel_size = 7, out_channels = 64, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout1 = QLinearLayerSign(np.prod(layer1.out_shape), output_neurons).to(device)

layer2 = DTNLIFConv2dLayer(inp_shape = layer1.out_shape, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 1, padding = 2, thr = thr, device = device).to(device)
random_readout2 = QLinearLayerSign(np.prod(layer2.out_shape), output_neurons).to(device)

layer3 = DTNLIFConv2dLayer(inp_shape = layer2.out_shape, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout3 = QLinearLayerSign(np.prod(layer3.out_shape), output_neurons).to(device)

layer4 = DTNLIFDenseLayer(in_channels = np.prod(layer3.out_shape), out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)

log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
nll_loss = torch.nn.NLLLoss()

# initlialize optimizier
opt1 = torch.optim.SGD(layer1.parameters(), lr=1)
opt2 = torch.optim.SGD(layer2.parameters(), lr=1)
opt3 = torch.optim.SGD(layer3.parameters(), lr=1)
opt4 = torch.optim.SGD(layer4.parameters(), lr=1)

train_acc = []
test_acc = []

print("WPQUEG Quantization: {0}{1}{2}{3}{4}{5}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb))

for e in range(50):
    correct = 0
    total = 0
    tcorrect = 0
    ttotal = 0
    loss_hist = []
    loss_hist2 = []
    loss_hist3 = []
    loss_hist4 = []
    start_time = time.time()

    for x_local, y_local in sparse_data_generator_DVS(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, shuffle = True, device = device):
        y_local = y_local -1
        class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)

        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])
        layer4.state_init(x_local.shape[0])

        # burnin
        for t in range(int(burnin/ms)):
            down_spikes = downsample(x_local[:,:,:,:,t])*16
            down_spikes[down_spikes < 0] = -1
            down_spikes[down_spikes > 0] = 1

            out_spikes1 = layer1.forward(down_spikes)
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)

        # training
        for t in range(int(burnin/ms), int(T/ms)):
            down_spikes = downsample(x_local[:,:,:,:,t])*16
            down_spikes[down_spikes < 0] = -1
            down_spikes[down_spikes > 0] = 1

            out_spikes1 = layer1.forward(down_spikes)
            rreadout1 = random_readout1(dropout_learning(DTNsmoothstep(layer1.U.reshape([x_local.shape[0], np.prod(layer1.out_shape)]))))
            y_log_p1 = log_softmax_fn(rreadout1)
            loss_t1 = nll_loss(y_log_p1, y_local) + lambda1 * F.relu(layer1.U+.01).mean() + lambda2 * F.relu(thr-layer1.U).mean()
            loss_t1.backward()
            opt1.step()
            opt1.zero_grad()

            out_spikes2 = layer2.forward(out_spikes1)
            rreadout2 = random_readout2(dropout_learning(DTNsmoothstep(layer2.U.reshape([x_local.shape[0], np.prod(layer2.out_shape)]))))
            y_log_p2 = log_softmax_fn(rreadout2)
            loss_t2 = nll_loss(y_log_p2, y_local) + lambda1 * F.relu(layer2.U+.01).mean() + lambda2 * F.relu(thr-layer2.U).mean()
            loss_t2.backward()
            opt2.step()
            opt2.zero_grad()

            out_spikes3 = layer3.forward(out_spikes2)
            rreadout3 = random_readout3(dropout_learning(DTNsmoothstep(layer3.U.reshape([x_local.shape[0], np.prod(layer3.out_shape)]))))
            y_log_p3 = log_softmax_fn(rreadout3)
            loss_t3 = nll_loss(y_log_p3, y_local) + lambda1 * F.relu(layer3.U+.01).mean() + lambda2 * F.relu(thr-layer3.U).mean()
            loss_t3.backward()
            opt3.step()
            opt3.zero_grad()

            # flattening for spiking readout layer
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)
            y_log_p4 = log_softmax_fn(DTNsmoothstep(layer4.U))
            loss_t4 = nll_loss(y_log_p4, y_local) + lambda1 * F.relu(layer4.U+.01).mean() + lambda2 * F.relu(.1-layer4.U).mean()
            loss_t4.backward()
            opt4.step()
            opt4.zero_grad()

            loss_hist.append(loss_t4.item())
            loss_hist2.append(loss_t3.item())
            loss_hist3.append(loss_t2.item())
            loss_hist4.append(loss_t1.item())
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

        for t in range(int(T_test/ms)):
            down_spikes = downsample(x_local[:,:,:,:,t])*16
            down_spikes[down_spikes < 0] = -1
            down_spikes[down_spikes > 0] = 1

            # dropout kept active -> decolle note
            out_spikes1 = dropout_learning(layer1.forward(down_spikes)) 
            out_spikes2 = dropout_learning(layer2.forward(out_spikes1)) 
            out_spikes3 = dropout_learning(layer3.forward(out_spikes2))
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = dropout_learning(layer4.forward(out_spikes3))
            class_rec += out_spikes4
        tcorrect += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        ttotal += len(y_local)
    inf_time = time.time()


    train_acc.append(correct.item()/total)
    test_acc.append(tcorrect.item()/ttotal)
    print("Epoch {0} | Loss: {1:.4f} Train Acc: {2:.4f} Test Acc: {3:.4f} Train Time: {4:.4f}s Inference Time: {5:.4f}s".format(e+1, np.mean(loss_hist), correct.item()/total, tcorrect.item()/ttotal, train_time-start_time, inf_time - train_time)) 


# saving results/weights
results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer3':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer4':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'test_acc': test_acc, 'train_acc': train_acc, 'loss':[loss_hist, loss_hist2, loss_hist3, loss_hist4]}
with open(args['dir'] + '/hello.pkl', 'wb') as f:
    pickle.dump(results, f)


#'train_idx':shuffle_idx_ta, 'test_idx':shuffle_idx_te