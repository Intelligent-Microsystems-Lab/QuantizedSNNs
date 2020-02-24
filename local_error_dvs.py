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
from localQ import sparse_data_generator_DVS, smoothstep, superspike, QLinearLayerSign, LIFDenseLayer, LIFConv2dLayer


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

layer1 = LIFConv2dLayer(inp_shape = (2, 32, 32), kernel_size = 7, out_channels = 64, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout1 = QLinearLayerSign(np.prod(layer1.out_shape), output_neurons).to(device)

layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 1, padding = 2, thr = thr, device = device).to(device)
random_readout2 = QLinearLayerSign(np.prod(layer2.out_shape), output_neurons).to(device)

layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout3 = QLinearLayerSign(np.prod(layer3.out_shape), output_neurons).to(device)

layer4 = LIFDenseLayer(in_channels = np.prod(layer3.out_shape), out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)

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

            # two channel trick
            down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
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

        # training
        for t in range(int(burnin/ms), int(T/ms)):
            down_spikes = downsample(x_local[:,:,:,:,t])*16

            # two channel trick
            down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
            mask1 = (down_spikes > 0) # this might change
            mask2 = (down_spikes < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            down_spikes = torch.zeros_like(down_spikes)
            down_spikes[mask1] = 1 
            down_spikes[mask2] = 1

            out_spikes1 = layer1.forward(down_spikes)
            rreadout1 = random_readout1(dropout_learning(smoothstep(layer1.U.reshape([x_local.shape[0], np.prod(layer1.out_shape)]))))
            y_log_p1 = log_softmax_fn(rreadout1)
            loss_t1 = nll_loss(y_log_p1, y_local) + lambda1 * F.relu(layer1.U+.01).mean() + lambda2 * F.relu(thr-layer1.U).mean()
            loss_t1.backward()
            opt1.step()
            opt1.zero_grad()

            out_spikes2 = layer2.forward(out_spikes1)
            rreadout2 = random_readout2(dropout_learning(smoothstep(layer2.U.reshape([x_local.shape[0], np.prod(layer2.out_shape)]))))
            y_log_p2 = log_softmax_fn(rreadout2)
            loss_t2 = nll_loss(y_log_p2, y_local) + lambda1 * F.relu(layer2.U+.01).mean() + lambda2 * F.relu(thr-layer2.U).mean()
            loss_t2.backward()
            opt2.step()
            opt2.zero_grad()

            out_spikes3 = layer3.forward(out_spikes2)
            rreadout3 = random_readout3(dropout_learning(smoothstep(layer3.U.reshape([x_local.shape[0], np.prod(layer3.out_shape)]))))
            y_log_p3 = log_softmax_fn(rreadout3)
            loss_t3 = nll_loss(y_log_p3, y_local) + lambda1 * F.relu(layer3.U+.01).mean() + lambda2 * F.relu(thr-layer3.U).mean()
            loss_t3.backward()
            opt3.step()
            opt3.zero_grad()

            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)
            y_log_p4 = log_softmax_fn(smoothstep(layer4.U))
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

            # two channel trick
            down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
            mask1 = (down_spikes > 0) # this might change
            mask2 = (down_spikes < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            down_spikes = torch.zeros_like(down_spikes)
            down_spikes[mask1] = 1 
            down_spikes[mask2] = 1

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
results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer3':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer4':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'test_acc': test_acc, 'train_acc': train_acc, 'loss':[loss_hist, loss_hist2, loss_hist3, loss_hist4], 'train_idx':shuffle_idx_ta, 'test_idx':shuffle_idx_te}
with open(args['dir'] + '/hello.pkl', 'wb') as f:
    pickle.dump(results, f)


