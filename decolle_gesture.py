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
from localQ import sparse_data_generator_Static, sparse_data_generator_DVSGesture, sparse_data_generator_DVSPoker, LIFConv2dLayer, prep_input, acc_comp, create_graph, DTNLIFConv2dLayer, create_graph2

import line_profiler

#torch.autograd.set_detect_anomaly(True)

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float32 # originally that was 64, but 32 is way faster
ms = 1e-3

# # DVS ASL
# ds_name = "DVS ASL"
# with open('data/dvs_asl.pickle', 'rb') as f:
#     data = pickle.load(f)

# data = np.array(data).T
# np.random.shuffle(data)
# split_point = int(data.shape[0]*.8)

# x_train = data[:split_point,0].tolist()
# y_train = data[:split_point,1].astype(np.int8)   - 1
# x_test = data[split_point:,0].tolist()
# y_test = data[split_point:,1].astype(np.int8)   - 1
# del data

# output_neurons = 24
# T = 100*ms
# T_test = 100*ms
# burnin = 10*ms
# x_size = 60
# y_size = 45


# # DVS Poker
# # load data
# ds_name = "DVS Poker"
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
# burnin = 50*ms
# x_size =
# y_size = 


# DVS Gesture
# load data
ds_name = "DVS Gesture"
with open('data/train_dvs_gesture88.pickle', 'rb') as f:
    data = pickle.load(f)
x_train = np.array(data[0])
y_train = np.array(data[1], dtype = int) - 1

idx_temp = np.arange(len(x_train))
np.random.shuffle(idx_temp)
idx_train = idx_temp[0:int(len(y_train)*.8)]
idx_val = idx_temp[int(len(y_train)*.8):]

x_train, x_val = x_train[idx_train], x_train[idx_val]
y_train, y_val = y_train[idx_train], y_train[idx_val]


with open('data/test_dvs_gesture88.pickle', 'rb') as f:
    data = pickle.load(f)
x_test = data[0]
y_test = np.array(data[1], dtype = int) - 1

output_neurons = 11
T = 500*ms
T_test = 1800*ms
burnin = 50*ms
x_size = 32
y_size = 32

change_diff = -2

# set quant level
quantization.global_wb  = 8
quantization.global_qb  = 10 
quantization.global_pb  = 12
quantization.global_rfb = 2

quantization.global_sb  = 6 
quantization.global_gb  = 10 + change_diff
quantization.global_eb  = 6 + change_diff

quantization.global_ub  = 6
quantization.global_ab  = 6
quantization.global_sig = 6

quantization.global_rb = 16
quantization.global_lr = 1#max([int(quantization.global_gb/8), 1]) if quantization.global_gb is not None else None
quantization.global_lr_sgd = 1.0e-9#np.geomspace(1.0e-2, 1.0e-9, 32)[quantization.global_wb-1]  if quantization.global_wb is not None else 1.0e-9
# quantization.global_lr_old = max([int(quantization.global_gb/8), 1]) if quantization.global_wb is not None else None # under development
quantization.global_beta = 1.5#quantization.step_d(quantization.global_wb)-.5 #1.5 #

# set parameters
delta_t = 1*ms
input_mode = 0
ds = 4 # downsampling

epochs = 320
lr_div = 60
batch_size = 72

PQ_cap = .75 #.75 #.1, .5, etc. # this value has to be carefully choosen
weight_mult = 4e-5#np.sqrt(4e-5) # decolle -> 1/p_max 
quantization.weight_mult = weight_mult

dropout_p = .5
localQ.lc_ampl = .5
# 0.0001, 0.001, 0.01, 0.1, .2, .5
l1 = .001
l2 = .001


tau_mem = torch.tensor([5*ms, 35*ms], dtype = dtype).to(device)#tau_mem = torch.tensor([5*ms, 35*ms], dtype = dtype).to(device)
tau_ref = torch.tensor([1/.35*ms], dtype = dtype).to(device)
tau_syn = torch.tensor([5*ms, 10*ms], dtype = dtype).to(device) #tau_syn = torch.tensor([5*ms, 10*ms], dtype = dtype).to(device)


sl1_loss = torch.nn.MSELoss()#torch.nn.SmoothL1Loss()

# # # construct layers
thr = torch.tensor([.0], dtype = dtype).to(device)
layer1 = LIFConv2dLayer(inp_shape = (2, x_size, y_size), kernel_size = 7, out_channels = 64, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = l1, l2 = l2, PQ_cap = PQ_cap, weight_mult = weight_mult, dtype = dtype).to(device)

layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape2, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 1, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = l1, l2 = l2, PQ_cap = PQ_cap, weight_mult = weight_mult, dtype = dtype).to(device)

layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape2, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = l1, l2 = l2, PQ_cap = PQ_cap, weight_mult = weight_mult, dtype = dtype).to(device)


# thr = torch.tensor([.4], dtype = dtype).to(device) 
# layer1 = DTNLIFConv2dLayer(inp_shape = (2, 32, 32), kernel_size = 7, out_channels = 64, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = l1, l2 = l2, PQ_cap = PQ_cap, weight_mult = weight_mult, dtype = dtype).to(device)

# layer2 = DTNLIFConv2dLayer(inp_shape = layer1.out_shape2, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 1, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = l1, l2 = l2, PQ_cap = PQ_cap, weight_mult = weight_mult, dtype = dtype).to(device)

# layer3 = DTNLIFConv2dLayer(inp_shape = layer2.out_shape2, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = l1, l2 = l2, PQ_cap = PQ_cap, weight_mult = weight_mult, dtype = dtype).to(device)

all_parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters())

# initlialize optimizier
if quantization.global_gb is not None:
    opt = torch.optim.SGD(all_parameters, lr=1)
else:
    opt = torch.optim.SGD(all_parameters, lr = quantization.global_lr_sgd) # 1.0e-9
    #opt = torch.optim.Adamax(all_parameters, lr=1.0e-9, betas=[0., .95])



def eval_test():
    batch_corr = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [], 'loss':[], 'act_train1':0, 'act_train2':0, 'act_train3':0, 'act_test1':0, 'act_test2':0, 'act_test3':0, 'w1u':0, 'w2u':0, 'w3u':0}
    # test accuracy
    for x_local, y_local in sparse_data_generator_DVSGesture(x_test, y_test, batch_size = batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True, ds = ds, x_size = x_size, y_size = y_size):
        rread_hist1_test = []
        rread_hist2_test = []
        rread_hist3_test = []
        act_spikes = [0,0,0]

        y_onehot = torch.Tensor(len(y_local), output_neurons).to(device).type(dtype)
        y_onehot.zero_()
        y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)


        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])

        for t in range(int(T_test/ms)):
            test_flag = (t > int(burnin/ms))

            out_spikes1, temp_loss1, temp_corr1, _ = layer1.forward(prep_input(x_local[:,:,:,:,t], input_mode), y_onehot, test_flag = test_flag)
            out_spikes2, temp_loss2, temp_corr2, _ = layer2.forward(out_spikes1, y_onehot, test_flag = test_flag)
            out_spikes3, temp_loss3, temp_corr3, _ = layer3.forward(out_spikes2, y_onehot, test_flag = test_flag)

            if test_flag:
                rread_hist1_test.append(temp_corr1)
                rread_hist2_test.append(temp_corr2)
                rread_hist3_test.append(temp_corr3)

            #batch_corr['act_test1'] += int(out_spikes1.sum())
            #batch_corr['act_test2'] += int(out_spikes2.sum())
            #batch_corr['act_test3'] += int(out_spikes3.sum())

        batch_corr['test1'].append(acc_comp(rread_hist1_test, y_local, True))
        batch_corr['test2'].append(acc_comp(rread_hist2_test, y_local, True))
        batch_corr['test3'].append(acc_comp(rread_hist3_test, y_local, True))
        #del x_local, y_local, y_onehot

    return torch.cat(batch_corr['test3']).mean()



args_compact = [delta_t, input_mode, ds, epochs, lr_div, batch_size, PQ_cap, weight_mult, dropout_p, localQ.lc_ampl, l1, l2, tau_mem, tau_ref, tau_syn, thr, quantization.global_wb, quantization.global_qb, quantization.global_pb, quantization.global_rfb, quantization.global_sb, quantization.global_gb, quantization.global_eb, quantization.global_ub, quantization.global_ab, quantization.global_sig, quantization.global_rb, quantization.global_lr, quantization.global_lr_sgd, quantization.global_beta]


diff_layers_acc = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [], 'loss':[], 'act_train1':[], 'act_train2':[], 'act_train3':[], 'act_test1':[], 'act_test2':[], 'act_test3':[], 'w1update':[], 'w2update':[], 'w3update':[]}
print("WUPQR SASigEG Quantization: {0}{1}{2}{3}{4} {5}{6}{7}{8}{9} l1 {10:.3f} l2 {11:.3f} Inp {12} LR {13} Drop {14} Cap {15} thr {16}".format(quantization.global_wb, quantization.global_ub, quantization.global_pb, quantization.global_qb, quantization.global_rfb, quantization.global_sb, quantization.global_ab, quantization.global_sig, quantization.global_eb, quantization.global_gb, l1, l2, input_mode, quantization.global_lr if quantization.global_lr != None else quantization.global_lr_sgd, dropout_p, PQ_cap, thr.item()))
plot_file_name = "DVS_WPQUEG{0}{1}{2}{3}{4}{5}{6}_Inp{7}_LR{8}_Drop{9}_thr{10}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb, quantization.global_sb, input_mode, quantization.global_lr, dropout_p, thr)+datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
print("Epoch Loss      Train1 Train2 Train3 Test1  Test2  Test3  | TrainT   TestT")

best_vali = torch.tensor(0, device = device)

for e in range(epochs):
    if ((e+1)%lr_div)==0:
        if quantization.global_gb is not None:
            quantization.global_lr /= 2
        else:
            opt.param_groups[-1]['lr'] /= 5


    batch_corr = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [], 'loss':[], 'act_train1':0, 'act_train2':0, 'act_train3':0, 'act_test1':0, 'act_test2':0, 'act_test3':0, 'w1u':0, 'w2u':0, 'w3u':0}
    quantization.global_w1update = 0 
    quantization.global_w2update = 0 
    quantization.global_w3update = 0 
    start_time = time.time()

    #P_rec = torch.tensor([])
    #Q_rec = torch.tensor([])
    # training
    for x_local, y_local in sparse_data_generator_DVSGesture(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, shuffle = True, test = False, device = device, ds = ds, x_size = x_size, y_size = y_size):

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

        #print("-------")

        for t in range(int(T/ms)):
            train_flag = (t > int(burnin/ms))

            out_spikes1, temp_loss1, temp_corr1, lparts1 = layer1.forward(prep_input(x_local[:,:,:,:,t], input_mode), y_onehot, train_flag = train_flag)
            out_spikes2, temp_loss2, temp_corr2, lparts2 = layer2.forward(out_spikes1, y_onehot, train_flag = train_flag)
            out_spikes3, temp_loss3, temp_corr3, lparts3 = layer3.forward(out_spikes2, y_onehot, train_flag = train_flag)
            

            #P_rec = torch.cat((P_rec, layer1.P.flatten().cpu()))
            #P_rec = torch.cat((P_rec, layer2.P.flatten().cpu()))
            #P_rec = torch.cat((P_rec, layer3.P.flatten().cpu()))

            #Q_rec = torch.cat((Q_rec, layer1.Q.flatten().cpu()))
            #Q_rec = torch.cat((Q_rec, layer2.Q.flatten().cpu()))
            #Q_rec = torch.cat((Q_rec, layer3.Q.flatten().cpu()))


            if train_flag:
                #print("{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f} {6:.4f} {7:.4f} {8:.4f}".format(lparts1[0].item(),lparts1[1].item(),lparts1[2].item(), lparts2[0].item(),lparts2[1].item(),lparts2[2].item(), lparts3[0].item(),lparts3[1].item(),lparts3[2].item()))
                loss_gen = temp_loss1 + temp_loss2 + temp_loss3

                loss_gen.backward()
                opt.step()
                opt.zero_grad()

                loss_hist.append(loss_gen.item())
                rread_hist1_train.append(temp_corr1)
                rread_hist2_train.append(temp_corr2)
                rread_hist3_train.append(temp_corr3)


            #batch_corr['act_train1'] += int(out_spikes1.sum())
            #batch_corr['act_train2'] += int(out_spikes2.sum())
            #batch_corr['act_train3'] += int(out_spikes3.sum())

        batch_corr['train1'].append(acc_comp(rread_hist1_train, y_local, True))
        batch_corr['train2'].append(acc_comp(rread_hist2_train, y_local, True))
        batch_corr['train3'].append(acc_comp(rread_hist3_train, y_local, True))
        del x_local, y_local, y_onehot

    train_time = time.time()

    diff_layers_acc['train1'].append(torch.cat(batch_corr['train1']).mean())
    diff_layers_acc['train2'].append(torch.cat(batch_corr['train2']).mean())
    diff_layers_acc['train3'].append(torch.cat(batch_corr['train3']).mean())
    #diff_layers_acc['act_train1'].append(batch_corr['act_train1'])
    #diff_layers_acc['act_train2'].append(batch_corr['act_train2'])
    #diff_layers_acc['act_train3'].append(batch_corr['act_train3'])
    diff_layers_acc['loss'].append(np.mean(loss_hist)/3)
    diff_layers_acc['w1update'].append(quantization.global_w1update)
    diff_layers_acc['w2update'].append(quantization.global_w2update)
    diff_layers_acc['w3update'].append(quantization.global_w3update)
        
    
    # test accuracy
    for x_local, y_local in sparse_data_generator_DVSGesture(x_val, y_val, batch_size = batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True, ds = ds, x_size = x_size, y_size = y_size):
        rread_hist1_test = []
        rread_hist2_test = []
        rread_hist3_test = []
        act_spikes = [0,0,0]

        y_onehot = torch.Tensor(len(y_local), output_neurons).to(device).type(dtype)
        y_onehot.zero_()
        y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)


        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])

        for t in range(int(T_test/ms)):
            test_flag = (t > int(burnin/ms))

            out_spikes1, temp_loss1, temp_corr1, _ = layer1.forward(prep_input(x_local[:,:,:,:,t], input_mode), y_onehot, test_flag = test_flag)
            out_spikes2, temp_loss2, temp_corr2, _ = layer2.forward(out_spikes1, y_onehot, test_flag = test_flag)
            out_spikes3, temp_loss3, temp_corr3, _ = layer3.forward(out_spikes2, y_onehot, test_flag = test_flag)

            if test_flag:
                rread_hist1_test.append(temp_corr1)
                rread_hist2_test.append(temp_corr2)
                rread_hist3_test.append(temp_corr3)

            #batch_corr['act_test1'] += int(out_spikes1.sum())
            #batch_corr['act_test2'] += int(out_spikes2.sum())
            #batch_corr['act_test3'] += int(out_spikes3.sum())

        batch_corr['test1'].append(acc_comp(rread_hist1_test, y_local, True))
        batch_corr['test2'].append(acc_comp(rread_hist2_test, y_local, True))
        batch_corr['test3'].append(acc_comp(rread_hist3_test, y_local, True))
        del x_local, y_local, y_onehot

    inf_time = time.time()

    if best_vali.item() < torch.cat(batch_corr['test3']).mean().item():
        best_vali = torch.cat(batch_corr['test3']).mean()
        test_acc_best_vali = eval_test()
        w1 = layer1.weights.data
        w2 = layer2.weights.data
        w3 = layer3.weights.data

    diff_layers_acc['test1'].append(torch.cat(batch_corr['test1']).mean())
    diff_layers_acc['test2'].append(torch.cat(batch_corr['test2']).mean())
    diff_layers_acc['test3'].append(torch.cat(batch_corr['test3']).mean())
    #diff_layers_acc['act_test1'].append(batch_corr['act_test1'])
    #diff_layers_acc['act_test2'].append(batch_corr['act_test2'])
    #diff_layers_acc['act_test3'].append(batch_corr['act_test3'])

    print("{0:02d}    {1:.3E} {2:.4f} {3:.4f} {4:.4f} {5:.4f} {6:.4f} {7:.4f} | {8:.4f} {9:.4f}".format(e+1, diff_layers_acc['loss'][-1], diff_layers_acc['train1'][-1], diff_layers_acc['train2'][-1], diff_layers_acc['train3'][-1], diff_layers_acc['test1'][-1], diff_layers_acc['test2'][-1], diff_layers_acc['test3'][-1], train_time - start_time, inf_time - train_time))
    #create_graph2(plot_file_name, diff_layers_acc, ds_name)
    create_graph(plot_file_name, diff_layers_acc, ds_name, test_acc_best_vali)


    # saving results
    results = {'acc': diff_layers_acc, 'fname':plot_file_name, 'args': args_compact}
    with open('results/'+plot_file_name+'.pkl', 'wb') as f:
        pickle.dump(results, f)

# saving results and weights
results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer2.weights.detach().cpu(), layer2.bias.detach().cpu()], 'layer3':[layer3.weights.detach().cpu(), layer3.bias.detach().cpu()], 'acc': diff_layers_acc, 'fname':plot_file_name, 'args': args_compact}
with open('results/'+plot_file_name+'.pkl', 'wb') as f:
    pickle.dump(results, f)


# # how to load
#with open("results/DVS_WPQUEG8121066106_Inp0_LR1_Drop0.5_20200501_010049.pkl", 'rb') as f:
#   # The protocol version used is detected automatically, so we do not
#   # have to specify it.
#   data = pickle.load(f)

#20200501_005657 - 0.0001
#20200501_005713 - 0.001
#20200501_005823 - 0.2
#20200501_010049 - 0.5

