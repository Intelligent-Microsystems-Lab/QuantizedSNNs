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
from localQ import sparse_data_generator_DVSGesture, onebatch_DVSGesture, sparse_data_generator_DVSPoker, LIFConv2dLayer, prep_input, acc_comp

torch.autograd.set_detect_anomaly(True)

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
burnin = 50*ms #40*ms
batch_size = 72
tau_ref = torch.Tensor([0*ms]).to(device)
dropout_p = .5
thr = torch.Tensor([0.]).to(device) #that probably should be one... one doesnt really work
localQ.lc_ampl = .5

l1 = .001#0.973#.5#0.5807472565567517#.5#0.485#
l2 = .001#1.099 #5#1.4068230901221566#.5#0.621#


tau_mem = torch.Tensor([5*ms, 35*ms]).to(device)#[mem_tau*ms-mem_tau*ms*var_perc, mem_tau*ms+mem_tau*ms*var_perc]
tau_ref = torch.Tensor([5*ms, 35*ms]).to(device)
tau_syn = torch.Tensor([5*ms, 10*ms]).to(device)#[syn_tau*ms-syn_tau*ms*var_perc, syn_tau*ms+syn_tau*ms*var_perc]

input_mode = 3 #two channel trick, down sample etc.

sig_fn = nn.Sigmoid()
log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
nll_loss = torch.nn.NLLLoss()
softmax_fn = nn.Softmax(dim=1)
sl1_loss = torch.nn.SmoothL1Loss()

# construct layers
downsample_l = nn.AvgPool2d(kernel_size = 4, stride = 4)

layer1 = LIFConv2dLayer(inp_shape = (2, 32, 32), kernel_size = 7, out_channels = 64, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, act = sig_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2, quant_on = quant_on).to(device)

layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape2, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 1, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, act = sig_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2, quant_on = quant_on).to(device)

layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape2, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, act = sig_fn, loss_fn = sl1_loss, l1 = l1, l2 = l2, quant_on = quant_on).to(device)


all_parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters())

# initlialize optimizier
if quant_on:
    opt = torch.optim.SGD(all_parameters, lr=1)
else:
    opt = torch.optim.Adamax(all_parameters, lr=1.0e-9, betas=[0., .95])


diff_layers_acc = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': []}

print("WPQUEG Quantization: {0}{1}{2}{3}{4}{5}{6} {7} l1 {8:.3f} l2 {9:.3f} Inp {10} LR {11} Drop {12}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb, quantization.global_sb, quant_on, l1, l2, input_mode, quantization.global_lr, dropout_p))




for e in range(epochs):
    if ((e+1)%1000)==0:
        opt.param_groups[-1]['lr']/=5

    rread_hist1_train = [] 
    rread_hist2_train = []
    rread_hist3_train = []

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
        train_flag = (t > int(burnin/ms))

        spikes_t                            = prep_input(x_local[:,:,:,:,t], input_mode)
        spikes_t                            = downsample_l(spikes_t)
        spikes_t[spikes_t > 0]              = 1
        spikes_t[spikes_t < 0]              = -1

        out_spikes1, temp_loss1, temp_corr1 = layer1.forward(spikes_t, y_onehot, train_flag = train_flag)
        out_spikes2, temp_loss2, temp_corr2 = layer2.forward(out_spikes1, y_onehot, train_flag = train_flag)
        out_spikes3, temp_loss3, temp_corr3 = layer3.forward(out_spikes2, y_onehot, train_flag = train_flag)
        if train_flag:
            loss_gen = temp_loss1 + temp_loss2 + temp_loss3 #+ temp_loss4

            loss_gen.backward()
            opt.step()
            opt.zero_grad()

            loss_hist.append(loss_gen.item())
            rread_hist1_train.append(temp_corr1)
            rread_hist2_train.append(temp_corr2)
            rread_hist3_train.append(temp_corr3)

    train_time = time.time()
 
    print("Epoch {0} | Loss: {1:.4f} Train Acc 1: {2:.4f} Train Acc 2: {3:.4f} Train Acc 3: {4:.4f} Train Time: {5:.4f}s".format(e+1, np.mean(loss_hist)/4, acc_comp(rread_hist1_train, y_local), acc_comp(rread_hist2_train, y_local), acc_comp(rread_hist3_train, y_local), train_time-start_time))
        
    
    # test accuracy
    if (e+1)%10 == 0:
        for x_local, y_local in sparse_data_generator_DVSGesture(x_test, y_test, batch_size = batch_size, nb_steps = T_test / ms, shuffle = False, device = device, test = True):
            rread_hist1_test = []
            rread_hist2_test = []
            rread_hist3_test = []

            y_onehot = torch.Tensor(len(y_local), output_neurons).to(device)
            y_onehot.zero_()
            y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)

            layer1.state_init(x_local.shape[0])
            layer2.state_init(x_local.shape[0])
            layer3.state_init(x_local.shape[0])

            for t in tqdm(range(int(T_test/ms))):
                test_flag = (t > int(burnin/ms))

                spikes_t                            = prep_input(x_local[:,:,:,:,t], input_mode)
                spikes_t                            = downsample_l(spikes_t)
                spikes_t[spikes_t > 0]              = 1
                spikes_t[spikes_t < 0]              = -1
                out_spikes1, temp_loss1, temp_corr1 = layer1.forward(spikes_t, y_onehot, test_flag = test_flag)
                out_spikes2, temp_loss2, temp_corr2 = layer2.forward(out_spikes1, y_onehot, test_flag = test_flag)
                out_spikes3, temp_loss3, temp_corr3 = layer3.forward(out_spikes2, y_onehot, test_flag = test_flag)

                if test_flag:
                    rread_hist1_test.append(temp_corr1)
                    rread_hist2_test.append(temp_corr2)
                    rread_hist3_test.append(temp_corr3)
        
            diff_layers_acc['test1'].append(acc_comp(rread_hist1_test, y_local))
            diff_layers_acc['test2'].append(acc_comp(rread_hist2_test, y_local))
            diff_layers_acc['test3'].append(acc_comp(rread_hist3_test, y_local))
            print("Test Acc 1: {0:.4f} Test Acc 2: {1:.4f} Test Acc 3: {2:.4f}".format( diff_layers_acc['test1'][-1], diff_layers_acc['test2'][-1], diff_layers_acc['test3'][-1]))


            
    #diff_layers_acc['train1'].append(correct1_train/total_train)
    #diff_layers_acc['train2'].append(correct2_train/total_train)
    #diff_layers_acc['train3'].append(correct3_train/total_train)
    



    

# saving results/weights
results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer3':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer4':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'loss':[loss_hist], 'acc': diff_layers_acc}


with open('results/'+str(uuid.uuid1())+'.pkl', 'wb') as f:
    pickle.dump(results, f)



