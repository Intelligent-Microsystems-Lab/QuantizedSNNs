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
import uuid

import quantization
from localQ import sparse_data_generator_DVS, sparse_data_generator_DVSPoker, smoothstep, superspike, QLinearLayerSign, LIFDenseLayer, LIFConv2dLayer, prep_input



class QSConv2dFunctional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, scale, padding = 0, pooling = None):
        w_quant = quantization.quant_w(weights, scale)
        bias_quant = quantization.quant_w(bias, scale)
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
        quant_error = quantization.quant_err(grad_output)

        # compute quantized error
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, w_quant, quant_error, padding = ctx.padding)
        # computed quantized gradient
        if ctx.needs_input_grad[1]:
            grad_weight = quantization.quant_grad(torch.nn.grad.conv2d_weight(input, w_quant.shape, quant_error, padding = ctx.padding)).float()
        # computed quantized bias
        if bias_quant is not None and ctx.needs_input_grad[2]:
            grad_bias = quantization.quant_grad(torch.einsum("abcd->b",(quant_error))).float()

        return grad_input, grad_weight, grad_bias, None, None, None


class LIFConv2dLayer(nn.Module):
    def __init__(self, inp_shape, kernel_size, out_channels, tau_syn, tau_mem, tau_ref, delta_t, pooling = 1, padding = 0, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float, dropout_p = .5, output_neurons = 10, loss_prep_fn = None, loss_fn = None, l1 = 0, l2 = 0):
        super(LIFConv2dLayer, self).__init__()   
        self.device = device
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.out_channels = out_channels  
        self.fan_in = kernel_size * kernel_size * inp_shape[0]
        self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_wb)]))
        self.L = np.max([np.sqrt( 6/self.fan_in), self.L_min])
        self.scale = 2 ** round(math.log(self.L_min / self.L, 2.0))
        self.scale = self.scale if self.scale > 1 else 1.0
        self.output_neurons = output_neurons

        self.padding = padding
        self.pooling = pooling
                
        self.dropout_learning = nn.Dropout(p=dropout_p)
        self.dropout_p = dropout_p
        self.l1 = l1
        self.l2 = l2
        self.loss_fn = loss_fn
        self.loss_prep_fn = loss_prep_fn

        self.weights = nn.Parameter(torch.empty((self.out_channels, inp_shape[0],  self.kernel_size, self.kernel_size),  device=device, dtype=dtype, requires_grad=True))
        torch.nn.init.uniform_(self.weights, a = -self.L, b = self.L)


        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype, requires_grad=True))
            torch.nn.init.uniform_(self.bias, a = -self.L, b = self.L)
        else:
            self.register_parameter('bias', None)

        self.out_shape = QSConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape).to(device), self.weights, self.bias, self.scale, self.padding, self.pooling).shape[1:]
        self.thr = thr

        self.sign_random_readout = QLinearLayerSign(np.prod(self.out_shape), output_neurons).to(device)

        if tau_syn.shape[0] == 2:
            self.beta = torch.exp( -delta_t / torch.Tensor(torch.Size(self.inp_shape)).uniform_(tau_syn[0], tau_syn[1]).to(device))
        else:
            self.beta = torch.Tensor([torch.exp( - delta_t / tau_syn)]).to(device)
        if tau_mem.shape[0] == 2:
            #2, 32, 32
            self.alpha = torch.exp( -delta_t / torch.Tensor(torch.Size(self.inp_shape)).uniform_(tau_mem[0], tau_mem[1]).to(device))
        else:
            self.alpha = torch.Tensor([torch.exp( - delta_t / tau_mem)]).to(device)

        if tau_ref.shape[0] == 2:
            self.gamma = torch.exp( -delta_t / torch.Tensor(torch.Size(self.out_shape)).uniform_(tau_ref[0], tau_ref[1]).to(device))
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

    
    def forward(self, input_t, y_local):
        with torch.no_grad():
            self.weights.data = quantization.clip(self.weights.data, quantization.global_gb)
            if self.bias is not None:
                self.bias.data = quantization.clip(self.bias.data, quantization.global_gb)

        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t

        # quantize P, Q
        self.P, _ = quantization.quant_generic(self.P, quantization.global_pb)
        self.Q, _ = quantization.quant_generic(self.Q, quantization.global_qb)

        self.U = QSConv2dFunctional.apply(self.P, self.weights, self.bias, self.scale, self.padding, self.pooling) + self.R

        # quantize U
        self.U, _ = quantization.quant_generic(self.U, quantization.global_ub)

        self.S = (self.U >= self.thr).float()

        rreadout = self.sign_random_readout(self.dropout_learning(smoothstep(self.U-self.thr).reshape([input_t.shape[0], np.prod(self.out_shape)])) * self.dropout_p)
        _, predicted = torch.max(rreadout.data, 1)

        if y_local.shape[1] == self.output_neurons:
            correct_train = (predicted == y_local.max(dim = 1 )[1]).sum().item()
        else:
            correct_train = (predicted == y_local).sum().item()
        loss_gen = self.loss_fn(self.loss_prep_fn(rreadout), y_local) + self.l1 * F.relu(self.U+.01).mean() + self.l2 * F.relu(self.thr-self.U).mean()

        return self.S, loss_gen, correct_train





# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float
verbose_output = True
ms = 1e-3
delta_t = 1*ms


# # DVS Poker
# # load data
# with open('../slow_poker_500_train.pickle', 'rb') as f:
#     data = pickle.load(f)
# x_train = data[0]
# y_train = data[1]

# with open('../slow_poker_500_test.pickle', 'rb') as f:
#     data = pickle.load(f)
# x_test = data[0]
# y_test = data[1]

# output_neurons = 4
# T = 500*ms
# T_test = 500*ms

# DVS Gesture
# load data
with open('data/small_train_dvs_gesture.pickle', 'rb') as f:
    data = pickle.load(f)
x_train = data[0]
y_train = np.array(data[1], dtype = int) - 1

with open('data/small_test_dvs_gesture.pickle', 'rb') as f:
    data = pickle.load(f)
x_test = data[0]
y_test = np.array(data[1], dtype = int) - 1

output_neurons = 11
T = 500*ms
T_test = 1800*ms


# set quant level
quantization.global_wb = 8
quantization.global_ub = 8
quantization.global_qb = 8
quantization.global_pb = 8
quantization.global_gb = 8
quantization.global_eb = 8
quantization.global_rb = 16
quantization.global_lr = 8
quantization.global_beta = 1.5 #quantization.step_d(quantization.global_wb)-.5

# set parameters
burnin = 50*ms
batch_size = 4#128
tau_ref = torch.Tensor([0*ms]).to(device)
dropout_p = .5
thr = torch.Tensor([.4]).to(device)

mem_tau = 19.144428947159064
syn_tau = 3.419011079385445
l1 = 0.5807472565567517
l2 = 1.4068230901221566
var_perc = 0.3797799366311833

tau_mem = torch.Tensor([mem_tau*ms-mem_tau*ms*var_perc, mem_tau*ms+mem_tau*ms*var_perc]).to(device)#torch.Tensor([5*ms, 35*ms]).to(device)
tau_syn = torch.Tensor([syn_tau*ms-syn_tau*ms*var_perc, syn_tau*ms+syn_tau*ms*var_perc]).to(device)#torch.Tensor([5*ms, 10*ms]).to(device)

input_mode = 0 #two channel trick, down sample etc.

log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
nll_loss = torch.nn.NLLLoss()
softmax_fn = nn.Softmax(dim=1)
sl1_loss = torch.nn.SmoothL1Loss()

# construct layers
downsample_l = nn.AvgPool2d(kernel_size = 4, stride = 4)

layer1 = LIFConv2dLayer(inp_shape = (2, 32, 32), kernel_size = 5, out_channels = 16, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_prep_fn = softmax_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2).to(device)

layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape, kernel_size = 5, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_prep_fn = softmax_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2).to(device)

layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape, kernel_size = 5, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_prep_fn = softmax_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2).to(device)

layer4 = LIFDenseLayer(in_channels = np.prod(layer3.out_shape), out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_prep_fn = softmax_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2).to(device)

all_parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters()) + list(layer4.parameters())

# initlialize optimizier
opt = torch.optim.SGD(all_parameters, lr=1)

train_acc = []
test_acc = []

print("WPQUEG Quantization: {0}{1}{2}{3}{4}{5} tau_mem {6:.2f} tau syn {7:.2f} l1 {8:.3f} l2 {9:.3f} var {10:.3f}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb, mem_tau, syn_tau, l1, l2, var_perc))


for e in range(75):
    if (e%20 == 0) and (e != 0) and (quantization.global_lr > 1):
        quantization.global_lr /= 2

    correct = 0
    total = 0
    tcorrect = 0
    ttotal = 0

    correct1_train = 0 
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

    for x_local, y_local in sparse_data_generator_DVS(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, shuffle = True, device = device):
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
            spikes_t          = prep_input(x_local[:,:,:,:,t], input_mode)
            spikes_t          = downsample_l(spikes_t)*16
            out_spikes1, _, _ = layer1.forward(spikes_t, y_onehot)
            out_spikes2, _, _ = layer2.forward(out_spikes1, y_onehot)
            out_spikes3, _, _ = layer3.forward(out_spikes2, y_onehot)
            out_spikes3       = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4, _, _ = layer4.forward(out_spikes3, y_onehot)

        # training
        for t in range(int(burnin/ms), int(T/ms)):
            spikes_t                            = prep_input(x_local[:,:,:,:,t], input_mode)
            spikes_t                            = downsample_l(spikes_t)*16
            out_spikes1, temp_loss1, temp_corr1 = layer1.forward(spikes_t, y_onehot)
            out_spikes2, temp_loss2, temp_corr2 = layer2.forward(out_spikes1, y_onehot)
            out_spikes3, temp_loss3, temp_corr3 = layer3.forward(out_spikes2, y_onehot)
            out_spikes3                         = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4, temp_loss4, temp_corr4 = layer4.forward(out_spikes3, y_onehot)

            loss_gen = temp_loss1 + temp_loss2 + temp_loss3 + temp_loss4

            loss_gen.backward()
            opt.step()
            opt.zero_grad()

            loss_hist.append(loss_gen.item())
            class_rec += out_spikes4
            correct1_train += temp_corr1
            correct2_train += temp_corr2
            correct3_train += temp_corr3
            correct4_train += temp_corr4
            total_train += y_local.size(0)


        correct += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        total += len(y_local)
    train_time = time.time()

    # test accuracy
    for x_local, y_local in sparse_data_generator_DVS(x_test, y_test, batch_size = batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True):
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
            spikes_t          = prep_input(x_local[:,:,:,:,t], input_mode)
            spikes_t          = downsample_l(spikes_t)*16
            out_spikes1, _, _ = layer1.forward(spikes_t, y_onehot)
            out_spikes2, _, _ = layer2.forward(out_spikes1, y_onehot)
            out_spikes3, _, _ = layer3.forward(out_spikes2, y_onehot)
            out_spikes3       = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4, _, _ = layer4.forward(out_spikes3, y_onehot)

        # testing
        for t in range(int(burnin/ms), int(T_test/ms)):
            spikes_t                            = prep_input(x_local[:,:,:,:,t], input_mode)
            spikes_t                            = downsample_l(spikes_t)*16
            out_spikes1, temp_loss1, temp_corr1 = layer1.forward(spikes_t, y_onehot)
            out_spikes2, temp_loss2, temp_corr2 = layer2.forward(out_spikes1, y_onehot)
            out_spikes3, temp_loss3, temp_corr3 = layer3.forward(out_spikes2, y_onehot)
            out_spikes3                         = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4, temp_loss4, temp_corr4 = layer4.forward(out_spikes3, y_onehot)

            
            class_rec += out_spikes4
            correct1_test += temp_corr1
            correct2_test += temp_corr2
            correct3_test += temp_corr3
            correct4_test += temp_corr4
            total_test += y_local.size(0)

        tcorrect += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        ttotal += len(y_local)
    inf_time = time.time()

    correct = correct.item()
    tcorrect = tcorrect.item()
    train_acc.append(correct/total)
    test_acc.append(tcorrect/ttotal)

    if verbose_output:
        print("Epoch {0} | Loss: {1:.4f} Train Acc 1: {2:.4f} Test Acc 1: {3:.4f} Train Acc 2: {4:.4f} Test Acc 2: {5:.4f} Train Acc 3: {6:.4f} Test Acc 3: {7:.4f} Train Acc 4: {8:.4f} Test Acc 4: {9:.4f}  TRAIN_ACC: {10:.4f} TEST_ACC: {11:.4f}  Train Time: {12:.4f}s Inference Time: {13:.4f}s".format(e+1, np.mean(loss_hist), correct1_train/total_train, correct1_test/total_test, correct2_train/total_train, correct2_test/total_test, correct3_train/total_train, correct3_test/total_test, correct4_train/total_train, correct4_test/total_test, correct/total, tcorrect/ttotal, train_time-start_time, inf_time - train_time))
    else:
        print("Epoch {0} | TRAIN_ACC: {1:.4f} TEST_ACC: {2:.4f}  Train Time: {3:.4f}s Inference Time: {4:.4f}s".format(e+1, correct/total, tcorrect/ttotal, train_time-start_time, inf_time - train_time))



# saving results/weights
results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer3':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer4':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'loss':[loss_hist], 'train': train_acc, 'test': test_acc}

with open('results/'+str(uuid.uuid1())+'.pkl', 'wb') as f:
    pickle.dump(results, f)


