import os
import datetime
from operator import itemgetter
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torchvision

from snn_loss import van_rossum

from superspike import SuperSpike, current2firing_time, sparse_data_generator 

from visual import mnist_train_curve, learning_para, precise_figs

from neurons import read_out_layer

from quantization import quantize

import line_profiler

#@profile
def get_weights(layers, device, time_step, tau_mem, scale_mult):
    dtype = torch.float
    beta = torch.exp(-time_step/tau_mem[0])
    weight_scale = scale_mult*(1.0-beta)
    weights = []

    for i,neurons in enumerate(layers):
        if i == 0:
            continue
        else:
            if 'convolutional' in list(layers)[i-1]:
                receving_neurons = int(np.sqrt(list(layers.values())[i]))-int(np.sqrt(list(layers.values())[i-1]))+1
                w1 = torch.empty((receving_neurons**2, list(layers.values())[i]), dtype=dtype, device=device, requires_grad=True)
            elif 'convolutional' in list(layers)[i]:
                w1 = torch.empty((list(layers.values())[i],1), dtype=dtype, device=device, requires_grad=True)
            else:
                w1 = torch.empty((list(layers.values())[i-1], list(layers.values())[i]), dtype=dtype, device=device, requires_grad=True)
            torch.nn.init.orthogonal_(w1, gain=weight_scale/np.sqrt(list(layers.values())[i]))
            #torch.nn.init.orthogonal_(w1, gain=weight_scale)
            weights.append(w1)

    return weights

#@profile
def run_snn_dropconnect(inputs, y, weights, layers, args, p_drop, infer):

    # normal layers
    for i, width in enumerate(layers):
        if i == 0:
            continue
        elif i == 1:
            with torch.no_grad():
                weights[-1] = quantize(weights[-1], nb=args['quant_nb'])
            _, spk_temp = args['neuron_type'](inputs=inputs, weights=weights[i-1], args = args, layer=i-1, layer_type = width, infer=infer)
        elif i < len(layers)-1:
            with torch.no_grad():
                weights[-1] = quantize(weights[-1], nb=args['quant_nb'])
            _, spk_temp = args['neuron_type'](inputs=spk_temp, weights=weights[i-1], args = args, layer=i-1, layer_type = width, infer=infer)
        else:
            continue

    # Readout layer
    with torch.no_grad():
        weights[-1] = quantize(weights[-1], nb=args['quant_nb'])
    m = read_out_layer(inputs = spk_temp, weights = weights[-1], args = args, infer = infer)
    return m

#@profile
def compute_classification_accuracy_dropconnect(x_data, y_data, weights, args_snn, layers):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    for x_local, y_local in args_snn['data_gen'](X = x_data, y =  y_data, batch_size = args_snn['batch_size'], nb_steps = args_snn['nb_steps'], nb_units = layers['input'], shuffle = True, time_step = args_snn['time_step'], device = args_snn['device']):
        m = run_snn_dropconnect(x_local.to_dense(), y_local, weights, layers, args_snn, args_snn['p_drop'], True)
        _,am=torch.max(m,1)
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
    return np.mean(accs)

#@profile
def train_classifier_dropconnect(x_data, y_data, x_test, y_test, nb_epochs, weights, args_snn, layers, figures, verbose, p_drop, fig_title="Training Curves"):
    optimizer = torch.optim.Adam(weights, lr=args_snn['lr'])
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()

    train_acc = []
    test_acc = []
    
    loss_hist = []
    print("Training: go")
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in args_snn['data_gen'](X = x_data, y =  y_data, batch_size = args_snn['batch_size'], nb_steps = args_snn['nb_steps'], nb_units = layers['input'], shuffle = True, time_step = args_snn['time_step'], device = args_snn['device']):

            with torch.autograd.detect_anomaly():
                optimizer.zero_grad()

                #weights = quantize(weights = weights)
                m = run_snn_dropconnect(x_local.to_dense(), y_local, weights, layers, args_snn, p_drop, False)
                log_p_y = log_softmax_fn(m)
                loss_val = loss_fn(log_p_y, y_local)

                loss_val.backward()
                optimizer.step()
                #weights = quantize(weights = weights, mu = args_snn['mu'], var = args_snn['var'])
            local_loss.append(float(loss_val.item()))
            if verbose:
                _,am=torch.max(m,1)
                print("Loss: "+str(loss_val)+" Predicted: "+str(am)+" Labels: "+str(y_local))
        #weights = quantize(weights = weights, mu = args_snn['mu'], var = args_snn['var'])
        #weights = quantize(weights = weights)
        train_acc.append(compute_classification_accuracy_dropconnect(x_data, y_data, weights, args_snn, layers))
        test_acc.append(compute_classification_accuracy_dropconnect(x_test, y_test, weights,args_snn, layers))

        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5e, train=%.5e, test=%.5e"%(e+1,mean_loss,train_acc[-1],test_acc[-1]))
        loss_hist.append(mean_loss)
        if figures:
            mnist_train_curve(loss_hist, train_acc, test_acc, fig_title, 'figures/results_'+args_snn['ds_name'] + "_" +args_snn['read_out']+"_" + args_snn['neuron_type'].__name__ + str('{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() ))+'.png')
            results = {'Parameters': args_snn, 'loss': loss_hist, 'train':train_acc, 'test': test_acc, 'w': weights}

            with open('results/results_'+args_snn['ds_name'] + "_" +args_snn['read_out']+"_" + args_snn['neuron_type'].__name__ + str('{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() ))+'.pkl', 'wb') as f:
                pickle.dump(results, f)
        if test_acc[-1] < 0.12:
            return loss_hist, train_acc, test_acc, weights

    return loss_hist, train_acc, test_acc, weights

#@profile
def gen_tau(mu=1e-3, var=1e-4, layers = [28*28, 500, 10], device = torch.device("cpu")):
    dtype = torch.float
    tau = []
    for i,layer_type in enumerate(layers):
        if i == 0:
            continue
        else:
            if 'convolutional' in layer_type:
                receving_neurons = int(np.sqrt(list(layers.values())[i-1]))-int(np.sqrt(list(layers.values())[i]))+1
                tau.append(torch.randn(receving_neurons**2, device = device, dtype = dtype)*var + mu)
            if 'fully-connected' in layer_type:
                tau.append(torch.randn(layers[layer_type], device = device, dtype = dtype)*var + mu)
            if 'output' in layer_type:
                tau.append(torch.randn(layers[layer_type], device = device, dtype = dtype)*var + mu)
    return tau




######### Following Code is not maintained anymore



def train_classifier(x_data, y_data, x_test, y_test, nb_epochs, weights, args_snn, layers, figures, verbose, fig_title="Training Curves"):
    optimizer = torch.optim.Adam(weights, lr=args_snn['lr'])

    train_acc = []
    test_acc = []
    
    loss_hist = []
    print("Training: go")
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in args_snn['data_gen'](X = x_data, y =  y_data, batch_size = args_snn['batch_size'], nb_steps = args_snn['nb_steps'], nb_units = len(weights[0]), shuffle = True, time_step = args_snn['time_step'], device = args_snn['device']):

            #with torch.autograd.detect_anomaly():
            loss_val, _, am = run_snn(x_local.to_dense(), y_local, weights, layers, args_snn)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
            if verbose:
                print("Loss: "+str(loss_val)+" Predicted: "+str(am)+" Labels: "+str(y_local))
            del x_local
            del y_local

        train_acc.append(compute_classification_accuracy(x_data,y_data, args_snn['batch_size'], args_snn['time_step'], args_snn['device'], args_snn['nb_steps'], len(weights[0]), run_snn, weights,args_snn, layers, args_snn['data_gen']))
        test_acc.append(compute_classification_accuracy(x_test,y_test, args_snn['batch_size'], args_snn['time_step'], args_snn['device'], args_snn['nb_steps'], len(weights[0]), run_snn,  weights,args_snn, layers, args_snn['data_gen']))
        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5e, train=%.5e, test=%.5e"%(e+1,mean_loss,train_acc[-1],test_acc[-1]))
        loss_hist.append(mean_loss)

    if figures:
        mnist_train_curve(loss_hist, train_acc, test_acc, fig_title)

    return loss_hist, train_acc, test_acc, weights

def training_precise(x_train, y_train, nb_epochs, weights, args_snn, layers, figures):

    optimizer = torch.optim.Adam(weights, lr=args_snn['lr'], betas=(0.9,0.999))

    local_loss = []

    for e in range(nb_epochs):
        #with torch.autograd.detect_anomaly():
        output,recs = run_snn(x_train, y_train, weights, layers, args_snn)
        loss_val = van_rossum(output, y_train, args_snn['time_step'], args_snn['tau_vr'], args_snn['device']) 

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        local_loss.append(loss_val.item())

        if e%10 == 0:
            print("Epoch %i: loss=%.5e"%(e,local_loss[-1]))

    if figures:
        precise_figs(y_train, output, local_loss, x_train)

    return local_loss, weights








def get_global_precise(x_train, y_train, var_name, init_value, final_value, beta, nb_epochs, weights, args_snn, figures, fig_title):

    mult = (final_value / init_value) ** (1/(nb_epochs-1))
    if var_name == 'lr':
        args_snn[var_name] = init_value
    else:
        args_snn[var_name] = torch.tensor([init_value], device=args_snn['device'], dtype=torch.float)
    optimizer = torch.optim.Adam(weights, lr=args_snn['lr'])

    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_tau = []


    optimizer = torch.optim.Adam(weights, lr=args_snn['lr'], betas=(0.9,0.999))

    local_loss = []

    for e in range(nb_epochs):
        one_step_losses = []
        for i in range(30):
            #with torch.autograd.detect_anomaly():
            output,recs = args_snn['run_snn'](x_train, weights, args_snn)
            loss_val = van_rossum(output, y_train, args_snn['time_step'], args_snn['tau_vr'], args_snn['device']) 
            one_step_losses.append(loss_val)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        loss_val = torch.mean(torch.stack(one_step_losses))

        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss_val
        smoothed_loss = avg_loss / (1 - beta**e)
        #Stop if the loss is exploding
        if e > 1 and smoothed_loss > 4 * best_loss:
            break
        #Record the best loss
        if smoothed_loss < best_loss or e==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        if var_name == 'lr':
            log_tau.append(np.log10(args_snn[var_name]))
        else:
            log_tau.append(torch.log10(args_snn[var_name]))


        local_loss.append(loss_val.item())

        if e%30 == 0:
            print(var_name+": %.5e, Loss: %.5e "%(args_snn[var_name], local_loss[-1]))

        args_snn[var_name] *= mult
        optimizer.param_groups[0]['lr'] = args_snn['lr']

    if figures:
        learning_para(log_tau, local_loss, losses, fig_title, var_name)

    return log_tau, losses, local_loss



def train_traditional_mnist(layers):
    # plain neural network implemented in pytorch
    print("not yet implemented")


def get_global(x_data, y_data, var_name, init_value, final_value, beta, weights, args_snn, figures, fig_title="Learning Rate"):

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    #learning: learning rate
    num = len(x_data)//args_snn['batch_size']
    mult = (final_value / init_value) ** (1/(num-1))
    if var_name == 'lr':
        args_snn[var_name] = init_value
    else:
        args_snn[var_name] = torch.tensor([init_value], device=args_snn['device'], dtype=torch.float)
    optimizer = torch.optim.Adam(weights, lr=args_snn['lr'])

    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_tau = []

    local_loss = []
    print(var_name + ": go")
    for x_local, y_local in sparse_data_generator(x_data, y_data, args_snn['batch_size'], args_snn['nb_steps'], len(weights[0]), True, args_snn['time_step'], args_snn['device']):
        #with torch.autograd.detect_anomaly():
        output,recs = args_snn['run_snn'](x_local.to_dense(), weights, args_snn)
        m,_=torch.max(output,1)
        log_p_y = log_softmax_fn(m)

        if args_snn['regularizer']:
            _,spks=recs
            reg_loss = 1e-5*torch.sum(spks) # L1 loss on total number of spikes
            reg_loss += 1e-5*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron

            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) + reg_loss
        else:
            loss_val = loss_fn(log_p_y, y_local)

        batch_num += 1

        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss_val
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        if var_name == 'lr':
            log_tau.append(np.log10(args_snn[var_name]))
        else:
            log_tau.append(torch.log10(args_snn[var_name]))

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        local_loss.append(loss_val.item())

        if batch_num%40 == 0:
            print(var_name+": %.5e, Loss: %.5e "%(args_snn[var_name], local_loss[-1]))

        args_snn[var_name] *= mult
        optimizer.param_groups[0]['lr'] = args_snn['lr']

    if figures:
        learning_para(log_tau, local_loss, losses, fig_title, var_name)

    return log_tau, losses, local_loss



def run_snn(inputs, y, weights, layers, args):
    mem_rec = []
    spk_rec = []

    for i, width in enumerate(layers):
        if i == 0:
            mem_temp, spk_temp = args['neuron_type'](inputs=inputs, weights=weights[i], args = args, layer = i, layer_type = width)
        elif i < len(layers)-2:
            mem_temp, spk_temp = args['neuron_type'](inputs=spk_rec[-1], weights=weights[i], args = args, layer = i, layer_type = width)
        else:
            continue
        mem_rec.append(mem_temp)
        spk_rec.append(spk_temp)

    # Readout layer
    if args['read_out'] != []:
        loss_val, out_rec, am = args['read_out'](inputs=spk_rec[-1], y = y, prev_spks = spk_rec[:-1], weights=weights[-1], args=args)

        other_recs = [mem_rec, spk_rec]
        return loss_val, other_recs, am
    
    return spk_rec[-1], mem_rec[-1]
