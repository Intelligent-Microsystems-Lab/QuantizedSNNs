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
from localQ import sparse_data_generator_Static, sparse_data_generator_DVSGesture, sparse_data_generator_DVSPoker, LIFConv2dLayer, prep_input, acc_comp, create_graph, hist_U_fun

import line_profiler

#torch.autograd.set_detect_anomaly(True)

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float32 # originally that was 64, but 32 is way faster


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
with open('data/train_dvs_gesture88.pickle', 'rb') as f:
    data = pickle.load(f)
x_train = data[0]
y_train = np.array(data[1], dtype = int) - 1

# x_train = x_train[:72]
# y_train = y_train[:72]


with open('data/test_dvs_gesture88.pickle', 'rb') as f:
    data = pickle.load(f)
x_test = data[0]
y_test = np.array(data[1], dtype = int) - 1

# x_test = x_test[:72]
# y_test = y_test[:72]

# set quant level
# base case 8 bits -> every variable gets its sweep (2,4,6,8)
quantization.global_wb  = 6 # what 2bits/2bit
quantization.global_ub  = 8 # 8 bits
quantization.global_qb  = 8 
quantization.global_pb  = 8
quantization.global_rfb = 8

quantization.global_sb  = 8 #4 bits
quantization.global_ab  = 8
quantization.global_sig = 8
quantization.global_gb  = 8
quantization.global_eb  = 8

quantization.global_rb = 16
quantization.global_lr = 1#max([int(quantization.global_gb/8), 1]) if quantization.global_gb is not None else None
quantization.global_lr_sgd = 1.0e-9#np.geomspace(1.0e-2, 1.0e-9, 32)[quantization.global_wb-1]  if quantization.global_wb is not None else 1.0e-9
# quantization.global_lr_old = max([int(quantization.global_gb/8), 1]) if quantization.global_wb is not None else None # under development
quantization.global_beta = 1.5#quantization.step_d(quantization.global_wb)-.5 #1.5 #

# set parameters
ms = 1e-3
delta_t = 1*ms
input_mode = -1
ds = 4 # downsampling

output_neurons = 11
T = 500*ms
T_test = 1800*ms
burnin = 50*ms
epochs = 320
lr_div = 60
batch_size = 72

PQ_cap = 1 #.1, .5, etc. # this value has to be carefully choosen
weight_mult = 4e-5#np.sqrt(4e-5) # decolle -> 1/p_max 
quantization.weight_mult = weight_mult

dropout_p = .5
localQ.lc_ampl = .5
l1 = .001
l2 = .001


thr = torch.tensor([0.], dtype = dtype).to(device) #that probably should be one... one doesnt really work
tau_mem = torch.tensor([5*ms, 35*ms], dtype = dtype).to(device)#tau_mem = torch.tensor([5*ms, 35*ms], dtype = dtype).to(device)
tau_ref = torch.tensor([1/.35*ms], dtype = dtype).to(device)
tau_syn = torch.tensor([5*ms, 10*ms], dtype = dtype).to(device) #tau_syn = torch.tensor([5*ms, 10*ms], dtype = dtype).to(device)

sl1_loss = torch.nn.MSELoss()#torch.nn.SmoothL1Loss()

# construct layers
layer1 = LIFConv2dLayer(inp_shape = (2, 32, 32), kernel_size = 7, out_channels = 64, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = l1, l2 = l2, PQ_cap = PQ_cap, weight_mult = weight_mult, dtype = dtype).to(device)

layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape2, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 1, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = l1, l2 = l2, PQ_cap = PQ_cap, weight_mult = weight_mult, dtype = dtype).to(device)

layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape2, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = l1, l2 = l2, PQ_cap = PQ_cap, weight_mult = weight_mult, dtype = dtype).to(device)


all_parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters())

# initlialize optimizier
if quantization.global_gb is not None:
    opt = torch.optim.SGD(all_parameters, lr=1)
else:
    opt = torch.optim.SGD(all_parameters, lr= quantization.global_lr_sgd) # 1.0e-9
    #opt = torch.optim.Adamax(all_parameters, lr=1.0e-9, betas=[0., .95])


diff_layers_acc = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [], 'loss':[]}

print("WUPQR SASigEG Quantization: {0}{1}{2}{3}{4} {5}{6}{7}{8}{9} l1 {10:.3f} l2 {11:.3f} Inp {12} LR {13} Drop {14} Cap {15}".format(quantization.global_wb, quantization.global_ub, quantization.global_pb, quantization.global_qb, quantization.global_rfb, quantization.global_sb, quantization.global_ab, quantization.global_sig, quantization.global_eb, quantization.global_gb, l1, l2, input_mode, quantization.global_lr if quantization.global_lr != None else quantization.global_lr_sgd, dropout_p, PQ_cap))

plot_file_name = "figures/DVS_WPQUEG{0}{1}{2}{3}{4}{5}{6}_Inp{7}_LR{8}_Drop_{9}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb, quantization.global_sb, input_mode, quantization.global_lr, dropout_p)+datetime.datetime.now().strftime("_%Y%m%d_%H%M%S") + ".png"

print("Epoch Loss      Train1 Train2 Train3 Test1  Test2  Test3  | TrainT   TestT")

for e in range(epochs):
    if ((e+1)%lr_div)==0:
        if quantization.global_gb is not None:
            quantization.global_lr /= 4
            if quantization.global_lr <= 1/16:
                quantization.global_lr = 1/16
        else:
            opt.param_groups[-1]['lr'] /= 5


    batch_corr = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [], 'loss':[]}
    start_time = time.time()

    # training
    for x_local, y_local in sparse_data_generator_DVSGesture(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, shuffle = True, test = False, device = device, ds = ds):

        y_onehot = torch.Tensor(len(y_local), output_neurons).to(device).type(dtype)
        y_onehot.zero_()
        y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)

        rread_hist1_train = [] 
        rread_hist2_train = []
        rread_hist3_train = []
        loss_hist = []

        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])


        for t in range(int(T/ms)):
            train_flag = (t > int(burnin/ms))

            out_spikes1, temp_loss1, temp_corr1 = layer1.forward(x_local[:,:,:,:,t], y_onehot, train_flag = train_flag)
            out_spikes2, temp_loss2, temp_corr2 = layer2.forward(out_spikes1, y_onehot, train_flag = train_flag)
            out_spikes3, temp_loss3, temp_corr3 = layer3.forward(out_spikes2, y_onehot, train_flag = train_flag)
            
            if train_flag:
                loss_gen = temp_loss1 + temp_loss2 + temp_loss3

                loss_gen.backward()
                opt.step()
                opt.zero_grad()

                loss_hist.append(loss_gen.item())
                rread_hist1_train.append(temp_corr1)
                rread_hist2_train.append(temp_corr2)
                rread_hist3_train.append(temp_corr3)



        batch_corr['train1'].append(acc_comp(rread_hist1_train, y_local, True))
        batch_corr['train2'].append(acc_comp(rread_hist2_train, y_local, True))
        batch_corr['train3'].append(acc_comp(rread_hist3_train, y_local, True))
        del x_local, y_local, y_onehot


    train_time = time.time()

    diff_layers_acc['train1'].append(torch.cat(batch_corr['train1']).mean())
    diff_layers_acc['train2'].append(torch.cat(batch_corr['train2']).mean())
    diff_layers_acc['train3'].append(torch.cat(batch_corr['train3']).mean())
    diff_layers_acc['loss'].append(np.mean(loss_hist)/4)
        
    
    # test accuracy
    for x_local, y_local in sparse_data_generator_DVSGesture(x_test, y_test, batch_size = batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True, ds = ds):
        rread_hist1_test = []
        rread_hist2_test = []
        rread_hist3_test = []

        y_onehot = torch.Tensor(len(y_local), output_neurons).to(device).type(dtype)
        y_onehot.zero_()
        y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)


        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])

        for t in range(int(T_test/ms)):
            test_flag = (t > int(burnin/ms))

            out_spikes1, temp_loss1, temp_corr1 = layer1.forward(x_local[:,:,:,:,t], y_onehot, test_flag = test_flag)
            out_spikes2, temp_loss2, temp_corr2 = layer2.forward(out_spikes1, y_onehot, test_flag = test_flag)
            out_spikes3, temp_loss3, temp_corr3 = layer3.forward(out_spikes2, y_onehot, test_flag = test_flag)

            if test_flag:
                rread_hist1_test.append(temp_corr1)
                rread_hist2_test.append(temp_corr2)
                rread_hist3_test.append(temp_corr3)

        batch_corr['test1'].append(acc_comp(rread_hist1_test, y_local, True))
        batch_corr['test2'].append(acc_comp(rread_hist2_test, y_local, True))
        batch_corr['test3'].append(acc_comp(rread_hist3_test, y_local, True))
        del x_local, y_local, y_onehot

    inf_time = time.time()

    diff_layers_acc['test1'].append(torch.cat(batch_corr['test1']).mean())
    diff_layers_acc['test2'].append(torch.cat(batch_corr['test2']).mean())
    diff_layers_acc['test3'].append(torch.cat(batch_corr['test3']).mean())

    print("{0:02d}    {1:.3E} {2:.4f} {3:.4f} {4:.4f} {5:.4f} {6:.4f} {7:.4f} | {8:.4f} {9:.4f}".format(e+1, diff_layers_acc['loss'][-1], diff_layers_acc['train1'][-1], diff_layers_acc['train2'][-1], diff_layers_acc['train3'][-1], diff_layers_acc['test1'][-1], diff_layers_acc['test2'][-1], diff_layers_acc['test3'][-1], train_time - start_time, inf_time - train_time))
    create_graph(plot_file_name, diff_layers_acc)


# saving results/weights
results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer2.weights.detach().cpu(), layer2.bias.detach().cpu()], 'layer3':[layer3.weights.detach().cpu(), layer3.bias.detach().cpu()], 'acc': diff_layers_acc, 'fname':plot_file_name}

with open('results/'+str(uuid.uuid1())+'.pkl', 'wb') as f:
    pickle.dump(results, f)


# how to load

#with open('results/40357372-7462-11ea-b0e2-a0369ffaa7c0.pkl', 'rb') as f:
#    # The protocol version used is detected automatically, so we do not
#    # have to specify it.
#    data = pickle.load(f)
