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
from tqdm import tqdm
import datetime
import uuid

import quantization
import localQ
from localQ import sparse_data_generator_DVSGesture, onebatch_DVSGesture, sparse_data_generator_DVSPoker, LIFConv2dLayer, prep_input


# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float
ms = 1e-3
delta_t = 1*ms
quant_on = False


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
with open('data/train_dvs_gesture.pickle', 'rb') as f:
    data = pickle.load(f)
x_train = data[0]
y_train = np.array(data[1], dtype = int) - 1

with open('data/test_dvs_gesture.pickle', 'rb') as f:
    data = pickle.load(f)
x_test = data[0]
y_test = np.array(data[1], dtype = int) - 1

output_neurons = 11
T = 500*ms
T_test = 1800*ms


# set quant level
quantization.global_wb = 2
quantization.global_ub = 2
quantization.global_qb = 2
quantization.global_pb = 2
quantization.global_gb = 2
quantization.global_eb = 2
quantization.global_rb = 2
quantization.global_lr = 1#8
quantization.global_sb = 1
quantization.global_beta = 1.5#quantization.step_d(quantization.global_wb)-.5 #1.5 #

# set parameters
epochs = 4500
burnin = 50*ms #50*ms
batch_size = 72
tau_ref = torch.Tensor([0*ms]).to(device)
dropout_p = .5
thr = torch.Tensor([0.]).to(device) #that probably should be one... one doesnt really work
localQ.lc_ampl = .5

l1 = .5#0.973#.5#0.5807472565567517#.5#0.485#
l2 = .5#1.099 #5#1.4068230901221566#.5#0.621#


tau_mem = torch.Tensor([5*ms, 35*ms]).to(device)#[mem_tau*ms-mem_tau*ms*var_perc, mem_tau*ms+mem_tau*ms*var_perc]
tau_ref = torch.Tensor([5*ms, 35*ms]).to(device)
tau_syn = torch.Tensor([5*ms, 10*ms]).to(device)#[syn_tau*ms-syn_tau*ms*var_perc, syn_tau*ms+syn_tau*ms*var_perc]

input_mode = 0 #two channel trick, down sample etc.

sig_fn = nn.Sigmoid()
log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
nll_loss = torch.nn.NLLLoss()
softmax_fn = nn.Softmax(dim=1)
sl1_loss = torch.nn.SmoothL1Loss()

# construct layers
downsample_l = nn.AvgPool2d(kernel_size = 4, stride = 4)

layer1 = LIFConv2dLayer(inp_shape = (2, 32, 32), kernel_size = 7, out_channels = 64, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_prep_fn = sig_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2, quant_on = quant_on).to(device)

layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 1, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_prep_fn = sig_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2, quant_on = quant_on).to(device)

layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_prep_fn = sig_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2, quant_on = quant_on).to(device)

#layer4 = LIFDenseLayer(in_channels = np.prod(layer3.out_shape), out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_prep_fn = softmax_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2).to(device)

all_parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters()) #+ list(layer4.parameters())

# initlialize optimizier
if quant_on:
    opt = torch.optim.SGD(all_parameters, lr=1)
else:
    opt = torch.optim.Adamax(all_parameters, lr=1.0e-9, betas=[0., .95])

#train_acc = []
#test_acc = []

diff_layers_acc = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': []}

print("WPQUEG Quantization: {0}{1}{2}{3}{4}{5}{6} {7} l1 {8:.3f} l2 {9:.3f} Inp {10} LR {11} Drop {12}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb, quantization.global_sb, quant_on, l1, l2, input_mode, quantization.global_lr, dropout_p))


for e in range(epochs):
    #if (e%20 == 0) and (e != 0) and (quantization.global_lr > 1):
    #    quantization.global_lr /= 2

    #correct = 0
    #total = 0
    #tcorrect = 0
    #ttotal = 0

    correct1_train = 0 
    correct2_train = 0
    correct3_train = 0
    #correct4_train = 0
    total_train = 0
    correct1_test = 0
    correct2_test = 0
    correct3_test = 0
    #correct4_test = 0
    total_test = 0
    loss_hist = []

    start_time = time.time()

    # training
    #for x_local, y_local in sparse_data_generator_DVSGesture(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, shuffle = True, device = device):

    x_local, y_local = onebatch_DVSGesture(x_train, y_train, batch_size = batch_size, nb_steps = T / ms,  device = device, shuffle = True)
    y_onehot = torch.Tensor(len(y_local), output_neurons).to(device)
    y_onehot.zero_()
    y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)

    layer1.state_init(x_local.shape[0])
    layer2.state_init(x_local.shape[0])
    layer3.state_init(x_local.shape[0])

    for t in tqdm(range(int(T/ms))):
        spikes_t                            = prep_input(x_local[:,:,:,:,t], input_mode)
        spikes_t                            = downsample_l(spikes_t)*16
        out_spikes1, temp_loss1, temp_corr1 = layer1.forward(spikes_t, y_onehot)
        out_spikes2, temp_loss2, temp_corr2 = layer2.forward(out_spikes1, y_onehot)
        out_spikes3, temp_loss3, temp_corr3 = layer3.forward(out_spikes2, y_onehot)
        if t > int(burnin/ms):

            loss_gen = temp_loss1 + temp_loss2 + temp_loss3 #+ temp_loss4

            loss_gen.backward()
            opt.step()
            opt.zero_grad()

            loss_hist.append(loss_gen.item())
            #class_rec += out_spikes4
            correct1_train += temp_corr1
            correct2_train += temp_corr2
            correct3_train += temp_corr3
            #correct4_train += temp_corr4
            total_train += y_local.size(0)

    train_time = time.time()
    print("Epoch {0} | Loss: {1:.4f} Train Acc 1: {2:.4f} Train Acc 2: {4:.4f} Train Acc 3: {6:.4f} Train Time: {8:.4f}s".format(e+1, np.mean(loss_hist), correct1_train/total_train, correct2_train/total_train, correct3_train/total_train, train_time-start_time))
        
    

    # test accuracy
    if (e+1)%10 == 0:
        for x_local, y_local in sparse_data_generator_DVSGesture(x_test, y_test, batch_size = batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True):
            y_onehot = torch.Tensor(len(y_local), output_neurons).to(device)
            y_onehot.zero_()
            y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)

            layer1.state_init(x_local.shape[0])
            layer2.state_init(x_local.shape[0])
            layer3.state_init(x_local.shape[0])

            for t in tqdm(range(int(T_test/ms))):
                spikes_t                            = prep_input(x_local[:,:,:,:,t], input_mode)
                spikes_t                            = downsample_l(spikes_t)*16
                out_spikes1, temp_loss1, temp_corr1 = layer1.forward(spikes_t, y_onehot)
                out_spikes2, temp_loss2, temp_corr2 = layer2.forward(out_spikes1, y_onehot)
                out_spikes3, temp_loss3, temp_corr3 = layer3.forward(out_spikes2, y_onehot)
                if t > int(burnin/ms):
                    correct1_test += temp_corr1
                    correct2_test += temp_corr2
                    correct3_test += temp_corr3
                    total_test += y_local.size(0)

        inf_time = time.time()
        diff_layers_acc['test1'].append(correct1_test/total_test)
        diff_layers_acc['test2'].append(correct2_test/total_test)
        diff_layers_acc['test3'].append(correct3_test/total_test)
        print("Test Acc 1: {0:.4f} Test Acc 2: {1:.4f} Test Acc 3: {2:.4f} Inf Time: {3:.4f}s".format( correct1_test/total_test, correct2_test/total_test, correct3_test/total_test, inf_time - train_time))


    diff_layers_acc['train1'].append(correct1_train/total_train)
    diff_layers_acc['train2'].append(correct2_train/total_train)
    diff_layers_acc['train3'].append(correct3_train/total_train)
    

    #correct = correct.item()
    #tcorrect = tcorrect.item()
    #train_acc.append(correct/total) WPQUEG: 1616161616161 l1 0.5 l2 0.5
    #test_acc.append(tcorrect/ttotal)


    

# saving results/weights
results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer3':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer4':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'loss':[loss_hist], 'train': train_acc, 'test': test_acc}

with open('results/'+str(uuid.uuid1())+'.pkl', 'wb') as f:
    pickle.dump(results, f)



