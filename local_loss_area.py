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
import h5py
import os


import quantization
import localQ
from localQ import sparse_data_generator_Static, sparse_data_generator_DVSGesture, sparse_data_generator_DVSPoker, LIFConv2dLayer, prep_input, acc_comp, create_graph, DTNLIFConv2dLayer, create_graph2

import line_profiler

# BaseLong.pkl, PQLong.pkl, EGLong.pkl, NoneLong.pkl
# PQShort.pkl, EGShort.pkl, NoneShort.pkl,

read_file = 'results/BaseLong.pkl'
surf_file = 'BaseLong.h5'
log_file = open('logs/BaseLong.log', 'w') 

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
T = 250*ms#500*ms
T_test = 1800*ms
burnin = 50*ms
x_size = 32
y_size = 32

# import argparse
# ap = argparse.ArgumentParser()
# ap.add_argument("-qp", "--qp", type = int, help = "weight bits")
# ap.add_argument("-s", "--s", type = int, help="multiplier")
# ap.add_argument("-eg", "--eg", type = int, help="dataset")
# args = vars(ap.parse_args())


change_diff1 = 0
change_diff2 = 0
change_diff3 = 0

# set quant level
quantization.global_wb  = 8
quantization.global_qb  = 10 + change_diff3
quantization.global_pb  = 12 + change_diff3
quantization.global_rfb = 2

quantization.global_sb  = 6 + change_diff2
quantization.global_gb  = 10 + change_diff1
quantization.global_eb  = 6 + change_diff1

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
batch_size = 236

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


# load weights
with open(read_file, 'rb') as f:
  # The protocol version used is detected automatically, so we do not
  # have to specify it.
  data = pickle.load(f)


layer1.weights.data = data['layer1'][2].to(device)
layer1.bias.data = data['layer1'][3].to(device)
layer2.weights.data = data['layer2'][2].to(device)
layer2.bias.data = data['layer2'][3].to(device)
layer3.weights.data = data['layer3'][2].to(device)
layer3.bias.data = data['layer3'][3].to(device)

layer1.sign_random_readout.weights.data = data['layer1'][4].to(device)
layer2.sign_random_readout.weights.data = data['layer2'][4].to(device)
layer3.sign_random_readout.weights.data = data['layer3'][4].to(device)

layer1.sign_random_readout.weight_fa.data = data['layer1'][5].to(device)
layer2.sign_random_readout.weight_fa.data = data['layer2'][5].to(device)
layer3.sign_random_readout.weight_fa.data = data['layer3'][5].to(device)


layer1.tau_mem = data['layer1'][6].to(device)
layer1.tau_syn = data['layer1'][7].to(device)
layer1.tau_ref = data['layer1'][8].to(device)
layer1.inp_mult_q = layer1.tau_syn
layer1.inp_mult_p = layer1.tau_mem
layer1.alpha = 1. -  (1. / layer1.tau_mem)
layer1.beta = 1. -  (1. / layer1.tau_syn)
layer1.gamma   = 1. - (1. / layer1.tau_ref)

layer2.tau_mem = data['layer2'][6].to(device)
layer2.tau_syn = data['layer2'][7].to(device)
layer2.tau_ref = data['layer2'][8].to(device)
layer2.inp_mult_q = layer2.tau_syn
layer2.inp_mult_p = layer2.tau_mem
layer2.alpha = 1. -  (1. / layer2.tau_mem)
layer2.beta = 1. -  (1. / layer2.tau_syn)
layer2.gamma   = 1. - (1. / layer2.tau_ref)

layer3.tau_mem = data['layer3'][6].to(device)
layer3.tau_syn = data['layer3'][7].to(device)
layer3.tau_ref = data['layer3'][8].to(device)
layer3.inp_mult_q = layer3.tau_syn
layer3.inp_mult_p = layer3.tau_mem
layer3.alpha = 1. -  (1. / layer3.tau_mem)
layer3.beta = 1. -  (1. / layer3.tau_syn)
layer3.gamma   = 1. - (1. / layer3.tau_ref)


layer1.Q_scale = (layer1.tau_syn/(1-layer1.beta)).max()
layer1.P_scale = ((layer1.tau_mem * layer1.Q_scale)/(1-layer1.alpha)).max()
layer1.Q_scale = (layer1.tau_syn/(1-layer1.beta)).max()

layer2.Q_scale = (layer2.tau_syn/(1-layer2.beta)).max()
layer2.P_scale = ((layer2.tau_mem * layer2.Q_scale)/(1-layer2.alpha)).max()
layer2.Q_scale = (layer2.tau_syn/(1-layer2.beta)).max()

layer3.Q_scale = (layer3.tau_syn/(1-layer3.beta)).max()
layer3.P_scale = ((layer3.tau_mem * layer3.Q_scale)/(1-layer3.alpha)).max()
layer3.Q_scale = (layer3.tau_syn/(1-layer3.beta)).max()

def eval_test():
    batch_corr = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [], 'loss':[], 'act_train1':0, 'act_train2':0, 'act_train3':0, 'act_test1':0, 'act_test2':0, 'act_test3':0, 'w1u':0, 'w2u':0, 'w3u':0}
    # test accuracy
    for x_local, y_local in sparse_data_generator_DVSGesture(x_test, y_test, batch_size = batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True, ds = ds, x_size = x_size, y_size = y_size):
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

#hello = eval_test()
#print("Expect Test: {0:.4f} Computed Test: {1:.4f}".format(hello.item(), data['evaled_test'].item()))


def eval_train_loss_acc():
    batch_corr = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [], 'loss':[], 'act_train1':0, 'act_train2':0, 'act_train3':0, 'act_test1':0, 'act_test2':0, 'act_test3':0, 'w1u':0, 'w2u':0, 'w3u':0}
    # test accuracy
    for x_local, y_local in sparse_data_generator_DVSGesture(x_val, y_val, batch_size = batch_size, nb_steps = T / ms, shuffle = True, device = device, test = False, ds = ds, x_size = x_size, y_size = y_size):
        rread_hist1_test = []
        rread_hist2_test = []
        rread_hist3_test = []
        loss_hist = [] 
        act_spikes = [0,0,0]

        y_onehot = torch.Tensor(len(y_local), output_neurons).to(device).type(dtype)
        y_onehot.zero_()
        y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)

        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])

        for t in range(int(T/ms)):
            test_flag = (t > int(burnin/ms))

            out_spikes1, temp_loss1, temp_corr1, lparts1 = layer1.forward(prep_input(x_local[:,:,:,:,t], input_mode), y_onehot, train_flag = test_flag)
            out_spikes2, temp_loss2, temp_corr2, lparts2 = layer2.forward(out_spikes1, y_onehot, train_flag = test_flag)
            out_spikes3, temp_loss3, temp_corr3, lparts3 = layer3.forward(out_spikes2, y_onehot, train_flag = test_flag)

            if test_flag:
                loss_gen = temp_loss1 + temp_loss2 + temp_loss3
                loss_hist.append(loss_gen.item())
                rread_hist1_test.append(temp_corr1)
                rread_hist2_test.append(temp_corr2)
                rread_hist3_test.append(temp_corr3)

        batch_corr['test1'].append(acc_comp(rread_hist1_test, y_local, True))
        batch_corr['test2'].append(acc_comp(rread_hist2_test, y_local, True))
        batch_corr['test3'].append(acc_comp(rread_hist3_test, y_local, True))

    #import pdb; pdb.set_trace()
    test3 = torch.cat(batch_corr['test3']).mean()
    over_loss = (np.mean(loss_hist)/3)/batch_size
    test_acc_best_vali = eval_test()

    return over_loss, test3




def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()) for w in weights]

def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def get_unplotted_indices(vals, xcoordinates, ycoordinates=None):
    """
    Args:
      vals: values at (x, y), with value -1 when the value is not yet calculated.
      xcoordinates: x locations, i.e.,[-1, -0.5, 0, 0.5, 1]
      ycoordinates: y locations, i.e.,[-1, -0.5, 0, 0.5, 1]

    Returns:
      - a list of indices into vals for points that have not yet been calculated.
      - a list of corresponding coordinates, with one x/y coordinate per row.
    """

    # Create a list of indices into the vectorizes vals
    inds = np.array(range(vals.size))

    # Select the indices of the un-recorded entries, assuming un-recorded entries
    # will be smaller than zero. In case some vals (other than loss values) are
    # negative and those indexces will be selected again and calcualted over and over.
    inds = inds[vals.ravel() <= 0]

    # Make lists containing the x- and y-coodinates of the points to be plotted
    if ycoordinates is not None:
        # If the plot is 2D, then use meshgrid to enumerate all coordinates in the 2D mesh
        xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
        s1 = xcoord_mesh.ravel()[inds]
        s2 = ycoord_mesh.ravel()[inds]
        return inds, np.c_[s1,s2]
    else:
        return inds, xcoordinates.ravel()[inds]



def set_weights(directions=None, step=None):
    dx = directions[0]
    dy = directions[1]
    changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]

    layer1.weights.data = (data['layer1'][2] + changes[0]).to(device)
    layer2.weights.data = (data['layer2'][2] + changes[1]).to(device)
    layer3.weights.data = (data['layer3'][2] + changes[2]).to(device)






ignore='biasbn' 
norm='filter'
weights = [layer1.weights.data, layer2.weights.data, layer3.weights.data]

xdirection = get_random_weights(weights)
normalize_directions_for_weights(xdirection, weights, norm, ignore)

ydirection = get_random_weights(weights)
normalize_directions_for_weights(ydirection, weights, norm, ignore)

directions = [xdirection, ydirection]

loss_key = 'train_loss'
acc_key = 'train_acc'


try:
    os.remove(surf_file)
except OSError:
    pass
f = h5py.File(surf_file, 'a')
f['dir_file'] = 'test_dir_file_loaded_in'

# Create the coordinates(resolutions) at which the function is evaluated
xcoordinates = np.linspace(-1, 1, num=25)
f['xcoordinates'] = xcoordinates
ycoordinates = np.linspace(-1, 1, num=25)
f['ycoordinates'] = ycoordinates
f.close()


f = h5py.File(surf_file, 'r+')
losses, accuracies = [], []
xcoordinates = f['xcoordinates'][:]
ycoordinates = f['ycoordinates'][:]


shape = (len(xcoordinates),len(ycoordinates))
losses = -np.ones(shape=shape)
accuracies = -np.ones(shape=shape)
f[loss_key] = losses
f[acc_key] = accuracies

inds, coords = get_unplotted_indices(losses, xcoordinates, ycoordinates)

print(read_file, file = log_file, flush=True)
print(surf_file, file = log_file, flush=True)
print("Start evaluating loss landscape:", file = log_file, flush=True)
print("We are in")

for count, ind in enumerate(inds):
    log_file.close()
    syc_start = time.time()
    coord = coords[count]
    set_weights(directions, coord)

    loss, acc = eval_train_loss_acc()
    losses.ravel()[ind] = loss
    accuracies.ravel()[ind] = acc

    f[loss_key][:] = losses
    f[acc_key][:] = accuracies
    f.flush()
    log_file = open('logs/baseT.log', 'w') 
    print("{0} evaled in {1:.4f}s: Loss {2:.4f} Acc {3:.4f}".format(count, time.time() - syc_start, loss, acc), file = log_file, flush=True)

log_file.close()
f.close()



