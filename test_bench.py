import os
import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torchvision

from neurons import LIF_neuron, adex_LIF_neuron, ferro_neuron, ferroLIF_neuron
from superspike import SuperSpike, sparse_data_generator, sparse_data_generator_DVS
from snn_training import train_classifier, get_weights, get_global, training_precise, get_global_precise, train_classifier_dropconnect, gen_tau
from visual import neuron_test


import line_profiler


dtype = torch.float
# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")


# One Dim Patter
# ds_name = "1D Pattern"
# x_train = torch.randn(1, 600, 500, device=device)
# x_train[x_train < 1.9] = 0
# x_train[x_train != 0] = 1
# y_train = torch.zeros(1, 600, 1, device=device)
# y_train[:,::20,0] = 1


# Two Dim Patter
# ds_name = "2D Pattern"
# x_train = torch.randn(1, 400, 500, device=device)
# x_train[x_train < 1.9] = 0
# x_train[x_train != 0] = 1
# y_train = torch.zeros(1, 400, 300, device=device)
# y_train[:,::50,::50] = 1


# ND
# ds_name = "ND"
# x_train_raw = pd.read_csv('data/ND/input.csv', sep=",", header=None)
# x_train = torch.zeros(1, 500, 500, device=device)
# x_train[:,x_train_raw[1],x_train_raw[0]] = 1
# y_train_raw = pd.read_csv('data/ND/target.csv', sep=",", header=None)
# y_train = torch.zeros(1, 500, 500, device=device)
# y_train[:,y_train_raw[1],y_train_raw[0]] = 1


# MNIST
ds_name = "MNIST"
train_dataset = torchvision.datasets.MNIST('../data/MNIST', train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.MNIST('../data/MNIST', train=False, transform=None, target_transform=None, download=True)
x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
x_train = x_train.reshape(x_train.shape[0],-1)/255
x_test = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
x_test = x_test.reshape(x_test.shape[0],-1)/255
y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)


# Fashion MNIST
# ds_name = "FMNIST"
# train_dataset = torchvision.datasets.FashionMNIST('../data/FashonMNIST', train=True, transform=None, target_transform=None, download=True)
# test_dataset = torchvision.datasets.FashionMNIST('../data/FashonMNIST', train=False, transform=None, target_transform=None, download=True)
# x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
# x_train = x_train.reshape(x_train.shape[0],-1)/255
# x_test  = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
# x_test  = x_test.reshape(x_test.shape[0],-1)/255
# y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
# y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)


# DVS
# ds_name = "DVS"
# test_dataset = pd.read_pickle('../DVS/test_complete.pkl')
# y_test = torch.tensor(test_dataset['label'], device=device, dtype=dtype)
# train_dataset = pd.read_pickle('../DVS/train_complete.pkl')
# y_train = torch.tensor(train_dataset['label'], device=device, dtype=dtype)
# with open('../DVS_prep/full_data_train.pkl', 'rb') as f:
#    train_data = pickle.load(f)
# with open('../DVS_prep/full_data_test.pkl', 'rb') as f:
#     test_data = pickle.load(f)
# x_test = pd.DataFrame({'batch':test_data[0],'ts':test_data[1],'unit':test_data[2]})
# x_test = x_test.drop_duplicates()
# x_train = pd.DataFrame({'batch':train_data[0],'ts':train_data[1],'unit':train_data[2]})
# x_train = x_train.drop_duplicates()


# parameters + architecture
layers = {'input'            : 28*28,
          #'convolutional_1'  : 5*5,
          'fully-connected_1': 100,
          'output'           : 10}
          #'input'            : 128*128,
          #'convolutional_1'  : 5*5,
          #'fully-connected_1': 8000,
          #'output'           : 12}


parameters = {
    # general 
    'ds_name'     : ds_name,
    'nb_epochs'   : 30,
    'neuron_type' : LIF_neuron,
    'read_out'    : "no_spike_integrate",
    'device'      : device,
    'dtype'       : torch.float,
    'spike_fn'    : SuperSpike.apply,
    'data_gen'    : sparse_data_generator,
    'time_step'   : 1e-3, #might need to be smaller
    'p_drop'      : 0.3,
    'batch_size'  : 64,
    'nb_steps'    : 100, #
    'lr'          : 5.58189e-04,
    'tau_vr'      : 4e-3,

    # LIF
    'fire_thresh' : gen_tau(mu = 1, var = 0, layers = layers, device = device),
    'tau_syn'     : gen_tau(mu = 5e-3, var = 0, layers = layers, device = device),
    'tau_mem'     : gen_tau(mu = 10e-4, var = 0, layers = layers, device = device),
   
    #adapt. exp.
    'tau_cur'     : gen_tau(mu = 2e-1, var = 0, layers = layers, device = device),
    'sharpness'   : gen_tau(mu = 0.04, var = 0, layers = layers, device = device),
    'a_cur'       : gen_tau(mu = 1e-40, var = 0, layers = layers, device = device),
    'b_cur'       : gen_tau(mu = 0.001, var = 0, layers = layers, device = device),
    'theta'       : gen_tau(mu = 0.95, var = 0, layers = layers, device = device),
    

    # ferro
    'v_rest_e'    : gen_tau(mu = -65 * 1e-3, var = 0, layers = layers, device = device),
    'dx_dt_param' : gen_tau(mu = 200e-3, var = 5e-5, layers = layers, device = device),
    'v_reset_e'   : gen_tau(mu = -65 * 1e-3, var = 0, layers = layers, device = device),
    'v_thresh_e'  : gen_tau(mu = -52*1e-3, var = 26e-5, layers = layers, device = device),
    'refrac_e'    : gen_tau(mu = 5*1e-3, var = 0, layers = layers, device = device),
    'tau_v'       : gen_tau(mu = 100 * 1e-3, var = 5e-4, layers = layers, device = device),
    'del_theta'   : gen_tau(mu = 0.1*1e-3, var = 0, layers = layers, device = device),
    'ge_max'      : gen_tau(mu = 8, var = 0, layers = layers, device = device),
    'gi_max'      : gen_tau(mu = 5, var = 0, layers = layers, device = device),
    'tau_ge'      : gen_tau(mu = 1*1e-3, var = 0, layers = layers, device = device),
    'tau_gi'      : gen_tau(mu = 2*1e-3, var = 0, layers = layers, device = device)
    }



weights = get_weights(layers, device=device, time_step=parameters['time_step'], tau_mem=parameters['tau_v'][0], scale_mult = 7)


loss_test, loss_train, train_acc, test_acc, weights = train_classifier_dropconnect(x_data = x_train, y_data = y_train, x_test = x_test, y_test = y_test, nb_epochs = parameters['nb_epochs'], weights = weights, args_snn = parameters, layers = layers, figures = True, verbose=False, p_drop = parameters['p_drop'],  fig_title=ds_name + " "+ parameters['read_out']+" "+ parameters['neuron_type'].__name__)


#results = {'Parameters': parameters, 'loss': loss_hist, 'train':train_acc, 'test': test_acc, 'w': result_w}

#with open('results/results_'+ds_name + "_" +parameters['read_out']+"_" + parameters['neuron_type'].__name__ + 'std_'+str(i*1e-5)+ str('{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() ))+'.pkl', 'wb') as f:
#    pickle.dump(results, f)

