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

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float
verbose_output = True

# load data
with open('../slow_poker_500_train.pickle', 'rb') as f:
    data = pickle.load(f)
x_train = data[0]
y_train = data[1]

with open('../slow_poker_500_test.pickle', 'rb') as f:
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
T_test = 500*ms
burnin = 50*ms
output_neurons = 4
batch_size = 128
tau_ref = torch.Tensor([0*ms]).to(device)
dropout_p = .5
thr = torch.Tensor([.4]).to(device)

mem_tau = 60 
syn_tau = 90
l1 =  1.35
l2 = .12
var_perc = .45


tau_mem = torch.Tensor([mem_tau*ms-mem_tau*ms*var_perc, mem_tau*ms+mem_tau*ms*var_perc]).to(device)#torch.Tensor([5*ms, 35*ms]).to(device)
tau_syn = torch.Tensor([syn_tau*ms-syn_tau*ms*var_perc, syn_tau*ms+syn_tau*ms*var_perc]).to(device)#torch.Tensor([5*ms, 10*ms]).to(device)
lambda1 = l1
lambda2 = l2

input_mode = 0 #two channel trick, down sample etc.

# construct layers
dropout_learning = nn.Dropout(p=dropout_p)

downsample_l = nn.AvgPool2d(kernel_size = 4, stride = 4)

layer1 = LIFConv2dLayer(inp_shape = (2, 32, 32), kernel_size = 5, out_channels = 16, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout1 = QLinearLayerSign(np.prod(layer1.out_shape), output_neurons).to(device)

layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape, kernel_size = 5, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout2 = QLinearLayerSign(np.prod(layer2.out_shape), output_neurons).to(device)

layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape, kernel_size = 5, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout3 = QLinearLayerSign(np.prod(layer3.out_shape), output_neurons).to(device)

layer4 = LIFDenseLayer(in_channels = np.prod(layer3.out_shape), out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)

log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
nll_loss = torch.nn.NLLLoss()
softmax_fn = nn.Softmax(dim=1)
sl1_loss = torch.nn.SmoothL1Loss()

all_parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters()) + list(layer4.parameters())

# initlialize optimizier
opt = torch.optim.SGD(all_parameters, lr=1)

train_acc = []
test_acc = []


print("WPQUEG Quantization: {0}{1}{2}{3}{4}{5} tau_mem {6:.2f} tau syn {7:.2f} l1 {8:.3f} l2 {9:.3f} var {9:.3f}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb, mem_tau, syn_tau, l1, l2, var_perc))


for e in range(45):
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

    rec_video = True
    for x_local, y_local in sparse_data_generator_DVSPoker(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, shuffle = True, device = device):

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
            spikes_t = prep_input(x_local[:,:,:,:,t], input_mode)
            spikes_t = downsample_l(spikes_t)*16
            out_spikes1 = layer1.forward(spikes_t)
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)

        # training
        for t in range(int(burnin/ms), int(T/ms)):
            total_train += y_local.size(0)
            loss_gen = 0

            spikes_t = prep_input(x_local[:,:,:,:,t], input_mode)
            spikes_t = downsample_l(spikes_t)*16
            out_spikes1 = layer1.forward(spikes_t)
            rreadout1 = random_readout1(dropout_learning( smoothstep(layer1.U-thr).reshape([x_local.shape[0], np.prod(layer1.out_shape)])) * dropout_p)
            #rreadout1 = random_readout1(dropout_learning(smoothstep(layer1.U.reshape([x_local.shape[0], np.prod(layer1.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct1_train += (predicted == y_local).sum().item()
            #import pdb; pdb.set_trace()
            loss_gen += sl1_loss(softmax_fn(rreadout1), y_onehot) + lambda1 * F.relu(layer1.U+.01).mean() + lambda2 * F.relu(thr-layer1.U).mean()
            #loss_gen += sl1_loss(((rreadout1 / rreadout1.abs().max())+1)*.5, y_onehot) #+ lambda1 * F.relu(layer1.U+.01).mean() + lambda2 * F.relu(thr-layer1.U).mean()

            out_spikes2 = layer2.forward(out_spikes1)
            rreadout2 = random_readout2(dropout_learning( smoothstep(layer2.U-thr).reshape([x_local.shape[0], np.prod(layer2.out_shape)])) * dropout_p)
            #rreadout2 = random_readout2(dropout_learning(smoothstep(layer2.U.reshape([x_local.shape[0], np.prod(layer2.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout2.data, 1)
            correct2_train += (predicted == y_local).sum().item()
            loss_gen += sl1_loss(softmax_fn(rreadout2), y_onehot) + lambda1 * F.relu(layer2.U+.01).mean() + lambda2 * F.relu(thr-layer2.U).mean()
            #loss_gen += sl1_loss( ((rreadout2 / rreadout2.abs().max())+1)*.5, y_onehot) #+ lambda1 * F.relu(layer2.U+.01).mean() + lambda2 * F.relu(thr-layer2.U).mean()

            out_spikes3 = layer3.forward(out_spikes2)
            rreadout3 = random_readout3(dropout_learning( smoothstep(layer3.U-thr).reshape([x_local.shape[0], np.prod(layer3.out_shape)])) * dropout_p)
            #rreadout3 = random_readout3(dropout_learning(smoothstep(layer3.U.reshape([x_local.shape[0], np.prod(layer3.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct3_train += (predicted == y_local).sum().item()
            loss_gen += sl1_loss(softmax_fn(rreadout3), y_onehot) + lambda1 * F.relu(layer3.U+.01).mean() + lambda2 * F.relu(thr-layer3.U).mean()
            #loss_gen += sl1_loss( ((rreadout3 / rreadout3.abs().max())+1)*.5, y_onehot) #+ lambda1 * F.relu(layer3.U+.01).mean() + lambda2 * F.relu(thr-layer3.U).mean()

            # flattening for spiking readout layer
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)
            loss_gen += sl1_loss(softmax_fn(smoothstep(layer4.U-thr)), y_onehot) + lambda1 * F.relu(layer4.U+.01).mean() + lambda2 * F.relu(thr-layer4.U).mean()
            #loss_gen += sl1_loss(smoothstep(layer4.U), y_onehot)
            _, predicted = torch.max(out_spikes4, 1)
            correct4_train += (predicted == y_local).sum().item() 
            #y_log_p4 = log_softmax_fn(smoothstep(layer4.U))
            #gen_loss +=  sl1_loss(smoothstep(layer4.U), y_onehot) + lambda1 * F.relu(layer4.U+.01).mean() + lambda2 * F.relu(.1-layer4.U).mean()

            loss_gen.backward()
            opt.step()
            opt.zero_grad()
            #print(loss_gen.item())

            loss_hist.append(loss_gen.item())
            class_rec += out_spikes4

        correct += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        total += len(y_local)
    train_time = time.time()


    # test accuracy
    for x_local, y_local in sparse_data_generator_DVSPoker(x_test, y_test, batch_size = batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True):
        y_local = y_local -1
        class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)
        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])
        layer4.state_init(x_local.shape[0])

        # burnin
        for t in range(int(burnin/ms)):
            spikes_t = prep_input(x_local[:,:,:,:,t], input_mode)
            spikes_t = downsample_l(spikes_t)*16
            out_spikes1 = layer1.forward(spikes_t)
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)

        # testing
        for t in range(int(burnin/ms), int(T_test/ms)):
            total_test += y_local.size(0)

            spikes_t = prep_input(x_local[:,:,:,:,t], input_mode)
            spikes_t = downsample_l(spikes_t)*16
            # dropout kept active -> decolle note
            out_spikes1 = layer1.forward(spikes_t)
            rreadout1 = random_readout1(dropout_learning(out_spikes1.reshape([x_local.shape[0], np.prod(layer1.out_shape)])) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct1_test += (predicted == y_local).sum().item()

            out_spikes2 = dropout_learning(layer2.forward(out_spikes1)) 
            rreadout2 = random_readout2(dropout_learning(out_spikes2.reshape([x_local.shape[0], np.prod(layer2.out_shape)])) * dropout_p)
            _, predicted = torch.max(rreadout2.data, 1)
            correct2_test += (predicted == y_local).sum().item()

            out_spikes3 = dropout_learning(layer3.forward(out_spikes2))
            rreadout3 = random_readout3(dropout_learning(out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])) * dropout_p)
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

    if verbose_output:
        print("Epoch {0} | Loss: {1:.4f} Train Acc 1: {2:.4f} Test Acc 1: {3:.4f} Train Acc 2: {4:.4f} Test Acc 2: {5:.4f} Train Acc 3: {6:.4f} Test Acc 3: {7:.4f} Train Acc 4: {8:.4f} Test Acc 4: {9:.4f}  TRAIN_ACC: {10:.4f} TEST_ACC: {11:.4f}  Train Time: {12:.4f}s Inference Time: {13:.4f}s".format(e+1, np.mean(loss_hist), correct1_train/total_train, correct1_test/total_train, correct2_train/total_train, correct2_test/total_train, correct3_train/total_train, correct3_test/total_train, correct4_train/total_train, correct4_test/total_train, correct/total, tcorrect/ttotal, train_time-start_time, inf_time - train_time))
    else:
        print("Epoch {0} | TRAIN_ACC: {1:.4f} TEST_ACC: {2:.4f}  Train Time: {3:.4f}s Inference Time: {4:.4f}s".format(e+1, correct/total, tcorrect/ttotal, train_time-start_time, inf_time - train_time))



# saving results/weights
results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer3':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer4':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'loss':[loss_hist], 'train': train_acc, 'test': test_acc}

with open('results/'+str(uuid.uuid1())+'.pkl', 'wb') as f:
    pickle.dump(results, f)

