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



def train_run(mem_tau, syn_tau, l1, l2, var_perc):

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

    diff_layers_acc = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [],'train4': [], 'test4': []}

    for e in range(2):
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
        for x_local, y_local in sparse_data_generator_DVSPoker(x_test, y_test, batch_size = batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True):
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
        diff_layers_acc['train1'].append(correct1_train/total_train)
        diff_layers_acc['test1'].append(correct1_test/total_test)
        diff_layers_acc['train2'].append(correct2_train/total_train)
        diff_layers_acc['test2'].append(correct2_test/total_test)
        diff_layers_acc['train3'].append(correct3_train/total_train)
        diff_layers_acc['test3'].append(correct3_test/total_test)
        diff_layers_acc['train4'].append(correct4_train/total_train)
        diff_layers_acc['test4'].append(correct4_test/total_test)

        if verbose_output:
            print("Epoch {0} | Loss: {1:.4f} Train Acc 1: {2:.4f} Test Acc 1: {3:.4f} Train Acc 2: {4:.4f} Test Acc 2: {5:.4f} Train Acc 3: {6:.4f} Test Acc 3: {7:.4f} Train Acc 4: {8:.4f} Test Acc 4: {9:.4f}  TRAIN_ACC: {10:.4f} TEST_ACC: {11:.4f}  Train Time: {12:.4f}s Inference Time: {13:.4f}s".format(e+1, np.mean(loss_hist), correct1_train/total_train, correct1_test/total_test, correct2_train/total_train, correct2_test/total_test, correct3_train/total_train, correct3_test/total_test, correct4_train/total_train, correct4_test/total_test, correct/total, tcorrect/ttotal, train_time-start_time, inf_time - train_time))
        else:
            print("Epoch {0} | TRAIN_ACC: {1:.4f} TEST_ACC: {2:.4f}  Train Time: {3:.4f}s Inference Time: {4:.4f}s".format(e+1, correct/total, tcorrect/ttotal, train_time-start_time, inf_time - train_time))



    return max(test_acc), {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer3':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer4':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'loss':[loss_hist], 'train': train_acc, 'test': test_acc, 'layers1':diff_layers_acc}

#best_test, res_dict = train_run(90, 90, 1.35, 1.12, .60)
#best_test, res_dict = train_run(60, 90, 1.35, .12, .45)
#print(best_test)

# saving results/weights
#results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer3':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer4':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'loss':[loss_hist]} # 'test_acc': test_acc, 'train_acc': train_acc, , 'train_idx':shuffle_idx_ta, 'test_idx':shuffle_idx_te
#with open('hello.pkl', 'wb') as f:
#    pickle.dump(results, f)


# Epoch 41 | Loss: 2.6689 Train Acc: 0.0816 Test Acc: 0.0833 Train Time: 734.5396s Inference Time: 298.9132s

from hyperopt import hp, fmin, tpe, space_eval

def objective(args):
    best_test, res_dict = train_run(args['mem_tau'], args['syn_tau'], args['l1'], args['l2'], args['var_perc'])
    #return 1-max(res_dict['layers1']['test4'])
    #return 1-max(res_dict['layers1']['test3'])
    return 1-best_test


space = {
    'mem_tau' : hp.uniform('mem_tau', 1, 130), 
    'syn_tau' : hp.uniform('syn_tau', 1, 130), 
    'l1' :      hp.uniform('l1', .05, 1.7),#1.3,#
    'l2' :      hp.uniform('l2', .05, 1.7),#0.15,#
    'var_perc' : hp.uniform('var_perc', 0, .9)
}

best = fmin(objective, space, algo=tpe.suggest, max_evals=75)
print(best)


#100%|██████████████████████████████████████████████| 75/75 [11:42:42<00:00, 562.16s/it, best loss: 0.7265625]
#{'l1': 0.0684335852531665,
#'l2': 1.1617847867426163,
#'mem_tau': 18.034992808871657,
#'syn_tau': 22.156518764463637}


# Exp:
# 1. all paras spike count opt      -
# 2. all paras last layer test      -
# 3. all paras last conv layer test - 
# 5. long run                       -
# 6. long run lr 8, 4, 2, 1         -