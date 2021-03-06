import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

import quantization


global lc_ampl
lc_ampl = .5

global shift_prob 
shift_prob = .5


def create_graph(plot_file_name, diff_layers_acc, ds_name, best_test):

    bit_string = str(quantization.global_wb) + str(quantization.global_ub) + str(quantization.global_pb) + str(quantization.global_qb) + str(quantization.global_rfb) + " " + str(quantization.global_sb) + str(quantization.global_ab) + str(quantization.global_sig) + str(quantization.global_eb) + str(quantization.global_gb)
    bit_string = bit_string.replace("None", "f")


    fig, ax1 = plt.subplots()
    fig.set_size_inches(8.4, 4.8)
    plt.title(ds_name + " "+ bit_string + " Test3: " + str(np.round( best_test.item(), 4)) + " " +str(shift_prob))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    t = np.arange(len(diff_layers_acc['loss']))
    ax1.plot(t, diff_layers_acc['train1'], 'g--', label = 'Train 1')
    ax1.plot(t, diff_layers_acc['train2'], 'b--', label = 'Train 2')
    ax1.plot(t, diff_layers_acc['train3'], 'r--', label = 'Train 3')
    ax1.plot(t, diff_layers_acc['test1'], 'g-', label = 'Test 1')
    ax1.plot(t, diff_layers_acc['test2'], 'b-', label = 'Test 2')
    ax1.plot(t, diff_layers_acc['test3'], 'r-', label = 'Test 3')
    ax1.plot([], [], 'k-', label = 'Loss')
    ax1.legend(bbox_to_anchor=(1.20,1), loc="upper left")
    #ax1.text(1.20, 0.1, str(max(diff_layers_acc['test3'])))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(t, diff_layers_acc['loss'], 'k-', label = 'Loss')

    fig.tight_layout()
    plt.savefig("figures/"+plot_file_name + ".png")
    plt.close()

def create_graph2(plot_file_name, diff_layers_acc, ds_name):

    bit_string = str(quantization.global_wb) + str(quantization.global_ub) + str(quantization.global_pb) + str(quantization.global_qb) + str(quantization.global_rfb) + " " + str(quantization.global_sb) + str(quantization.global_ab) + str(quantization.global_sig) + str(quantization.global_eb) + str(quantization.global_gb)
    bit_string = bit_string.replace("None", "f")


    fig, ax1 = plt.subplots()
    fig.set_size_inches(8.4, 4.8)
    plt.title(ds_name + " Act "+ bit_string)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('# Spikes/Updates')
    t = np.arange(len(diff_layers_acc['loss']))

    ax1.plot(t, diff_layers_acc['act_train1'], 'g--', label = 'Train 1')
    ax1.plot(t, diff_layers_acc['act_train2'], 'b--', label = 'Train 2')
    ax1.plot(t, diff_layers_acc['act_train3'], 'r--', label = 'Train 3')
    ax1.plot(t, diff_layers_acc['act_test1'], 'g-', label = 'Test 1')
    ax1.plot(t, diff_layers_acc['act_test2'], 'b-', label = 'Test 2')
    ax1.plot(t, diff_layers_acc['act_test3'], 'r-', label = 'Test 3')
    ax1.plot(t, diff_layers_acc['w1update'], 'm-', label = 'W update 1')
    ax1.plot(t, diff_layers_acc['w2update'], 'k-', label = 'W update 2')
    ax1.plot(t, diff_layers_acc['w3update'], 'y-', label = 'W update 3')
    ax1.legend(bbox_to_anchor=(1.20,1), loc="upper left")
    #ax1.text(1.20, 0.1, str(max(diff_layers_acc['test3'])))

    #ax2 = ax1.twinx()
    #ax2.set_ylabel('Loss')
    #ax2.plot(t, diff_layers_acc['loss'], 'k-', label = 'Loss')

    fig.tight_layout()
    plt.savefig("figures/"+plot_file_name+ "_act.png")
    plt.close()



def acc_comp(rread_hist_train, y_local, bools = False):
    rhts = torch.stack(rread_hist_train, dim = 0)
    if bools:
        return (rhts.mode(0)[0] == y_local).float()
    return (rhts.mode(0)[0] == y_local).float().mean()

def clee_spikes(T, rates):
    spikes = np.ones((T, + np.prod(rates.shape)))        
    spikes[np.random.binomial(1, (1000. - rates.flatten())/1000, size=(T, np.prod(rates.shape))).astype('bool')] = 0
    return spikes.T.reshape((rates.shape + (T,)))

def prep_input(x_local, input_mode, channels = 2):
    #two channel trick / decolle
    if input_mode == 0:
        x_local[x_local > 0] = 1

        #down_spikes = torch.cat((x_local, x_local), dim = 1)
        #mask1 = (down_spikes > 0) # this might change
        #mask2 = (down_spikes < 0)
        #mask1[:,0,:,:] = False
        #mask2[:,1,:,:] = False
        #down_spikes = torch.zeros_like(down_spikes)
        #down_spikes[mask1] = 1 
        #down_spikes[mask2] = 1
        return x_local
    #bi directional
    if input_mode == 2:
        x_local[:,0,:,:] *= -1
        new_spikes = x_local[:,0,:,:] + x_local[:,1,:,:]
        new_spikes = new_spikes.reshape([x_local.shape[0], 1, x_local.shape[2], x_local.shape[3]])
        new_spikes[new_spikes > 0] = 1
        new_spikes[new_spikes < 0] = -1
        return new_spikes
    # same same but different
    if input_mode == 1:
        down_spikes = x_local
        down_spikes[down_spikes != 0] = 1
        return down_spikes
    #bi directional two channels
    if input_mode == 3:
        x_local[:,0,:,:] *= -1
        new_spikes = x_local[:,0,:,:] + x_local[:,1,:,:]
        new_spikes = new_spikes.reshape([x_local.shape[0], 1, x_local.shape[2], x_local.shape[3]])
        new_spikes[new_spikes > 0] = 1
        new_spikes[new_spikes < 0] = -1

        new_spikes = torch.cat((new_spikes, new_spikes), dim = 1)
        return new_spikes
    else:
        return x_local


def sparse_data_generator_DVSPoker(X, y, batch_size, nb_steps, shuffle, device, test = False):
    number_of_batches = int(np.ceil(len(y)/batch_size))
    sample_index = np.arange(len(y))
    nb_steps = nb_steps -1
    y = np.array(y)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        all_events = np.array([[],[],[],[],[],[],[]]).T


        for bc,idx in enumerate(batch_index):
            temp = np.append(np.ones((X[idx].shape[0], 1))*bc, X[idx], axis=1)
            all_events = np.append(all_events, temp, axis = 0)

        # to matrix
        all_events = all_events[:,[0,4,5,1,6]]
        sparse_matrix = torch.sparse.FloatTensor(torch.LongTensor(all_events[:,[True, True, True, True, False]].T), torch.FloatTensor(all_events[:,4])).to_dense()

        # quick trick...
        sparse_matrix[sparse_matrix < 0] = -1
        sparse_matrix[sparse_matrix > 0] = 1

        sparse_matrix = sparse_matrix.reshape(torch.Size([sparse_matrix.shape[0], 1, sparse_matrix.shape[1], sparse_matrix.shape[2], sparse_matrix.shape[3]]))

        y_batch = torch.tensor(y[batch_index], dtype = int)
        try:
            torch.cuda.empty_cache()
            yield sparse_matrix.to(device=device), y_batch.to(device=device)
            counter += 1
        except StopIteration:
            return

def sparse_data_generator_DVSGesture(X, y, batch_size, nb_steps, shuffle, device, ds = 4, test = False, x_size = 32, y_size = 32):
    number_of_batches = int(np.ceil(len(y)/batch_size))
    sample_index = np.arange(len(y))
    nb_steps = nb_steps -1
    y = np.array(y)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        all_events = np.array([[],[],[],[],[]]).T

        y_batch = torch.tensor(y[batch_index], dtype = int)

        for bc,idx in enumerate(batch_index):
            # stochasticity here -> we sample
            if test:
                start_ts = 0
            else:
                start_ts = np.random.choice(np.arange(np.max(X[idx][:,0]) - nb_steps),1)
            temp = X[idx][X[idx][:,0] >= start_ts]
            temp = temp[temp[:,0] <= start_ts+nb_steps]
            temp = np.append(np.ones((temp.shape[0], 1))*bc, temp, axis=1)
            temp[:,1] = temp[:,1] - start_ts
            all_events = np.append(all_events, temp, axis = 0)

        # to matrix
        #all_events[:,4][all_events[:,4] == 0] = -1
        # spike_ind = (x_local == 1).nonzero()
        # spike_ind = spike_ind[torch.bernoulli((.5) * torch.ones(spike_ind.shape[0])).bool()]
        # spike_ind = spike_ind[torch.randperm(spike_ind.shape[0])]
        # split_point = int(spike_ind.shape[0]/2)
        # forward_spike = spike_ind[0:split_point]
        # backward_spike = spike_ind[split_point:]

        # x_local[torch.sparse.FloatTensor(forward_spike.t(), torch.ones(forward_spike.shape[0]).to(device)).to_dense().bool()] = 0
        # forward_spike[:,4] =  forward_spike[:,4] + 1
        # forward_spike[forward_spike[:,4] == 500] = 499
        # x_local[torch.sparse.FloatTensor(forward_spike.t(), torch.ones(forward_spike.shape[0]).to(device)).to_dense().bool()] = 1

        # x_local[torch.sparse.FloatTensor(backward_spike.t(), torch.ones(backward_spike.shape[0]).to(device)).to_dense().bool()] = 0
        # backward_spike[:,4] =  backward_spike[:,4] - 1
        # backward_spike[backward_spike[:,4] == -1] = 0
        # x_local[torch.sparse.FloatTensor(backward_spike.t(), torch.ones(backward_spike.shape[0]).to(device)).to_dense().bool()] = 1

        
        #change
        # by plus minus one process...
        # change_mask = torch.bernoulli((shift_prob) * torch.ones(all_events.shape[0])).bool()
        # forward_mask = change_mask * torch.bernoulli((.5) * torch.ones(all_events.shape[0])).bool()
        # backward_mask = (change_mask != forward_mask)
        # all_events[forward_mask, 1] = all_events[forward_mask, 1] + 1 #torch.randn(all_events[forward_mask, 1].shape[0])
        # all_events[backward_mask, 1] = all_events[backward_mask, 1] - 1

        all_events[:, 1] = all_events[:, 1] + (shift_prob*np.random.randn(all_events[:, 1].shape[0])).astype(int)

        neg_ind = (all_events[:,1] < 0)
        pos_ind = (all_events[:,1] > nb_steps)
        all_events[neg_ind,1] = 0  
        all_events[pos_ind,1] = int(nb_steps)


        all_events = all_events[:,[0,4,2,3,1]]
        all_events[:, 2] = all_events[:, 2]//ds
        all_events[:, 3] = all_events[:, 3]//ds
        sparse_matrix = torch.sparse.FloatTensor(torch.LongTensor(all_events[:,[True, True, True, True, True]].T), torch.ones_like(torch.tensor(all_events[:,0])), torch.Size([len(y_batch),2,x_size,y_size,int(nb_steps+1)])).to_dense().type(torch.int16)

        # quick trick...
        #sparse_matrix[sparse_matrix != 0] = 1
        #sparse_matrix[sparse_matrix > 0] = 1
        #sparse_matrix = sparse_matrix.reshape(torch.Size([sparse_matrix.shape[0], 1, sparse_matrix.shape[1], sparse_matrix.shape[2], sparse_matrix.shape[3]]))

        
        try:
            torch.cuda.empty_cache()
            yield sparse_matrix.to(device=device), y_batch.to(device=device)
            counter += 1
        except StopIteration:
            return

def sparse_data_generator_Static(X, y, batch_size, nb_steps, samples, max_hertz, shuffle=True, device=torch.device("cpu")):
    sample_idx = torch.randperm(len(X))[:samples]
    number_of_batches = int(np.ceil(samples/batch_size))
    nb_steps = int(nb_steps)

    counter = 0
    while counter<number_of_batches:
        if counter == number_of_batches:
            cur_sample = sample_idx[batch_size*counter:]
        else:
            cur_sample = sample_idx[batch_size*counter:batch_size*(counter+1)]

        X_batch = np.zeros((cur_sample.shape[0],) + X.shape[1:] + (nb_steps,))
        for i,idx in enumerate(cur_sample):
            X_batch[i] = clee_spikes(T = nb_steps, rates=max_hertz*X[idx,:]).astype(np.float32)

        X_batch = torch.from_numpy(X_batch).float()
        y_batch = y[cur_sample]
        try:
            torch.cuda.empty_cache()
            yield X_batch.to(device), y_batch.to(device)
            counter += 1
        except StopIteration:
            return

class quantU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return quantization.quant11(torch.clamp(x/8, -8, 8), quantization.global_ub)*8

    @staticmethod
    def backward(ctx, grad_output):     
        return grad_output



class QLinearFunctional(torch.autograd.Function):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    @staticmethod
    def forward(ctx, input, weight, weight_fa, bias, scale):
        if quantization.global_sig is not None:
            input = quantization.quant_sig(input)

        output = torch.einsum('ab,cb->ac', input, weight)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        if quantization.global_sb is not None:
            output = output/scale
        # quant act
        if quantization.global_ab is not None:
            output, _ = quantization.quant_act(output)

        ctx.save_for_backward(input, weight, weight_fa, bias)

        # ify part here... shall we bring it between 0 and 1 for the targets
        return (output+1)/2

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_fa, bias = ctx.saved_tensors
        grad_input = None

        if quantization.global_eb is not None:
            quant_error = quantization.quant_err(grad_output) #* clip_info.float()
        else:
            quant_error = grad_output

        if ctx.needs_input_grad[0]:
            grad_input = torch.einsum('ab,bc->ac', quant_error, weight_fa)

        # quantizing here for sigmoid input
        if quantization.global_eb is not None:
            grad_input = quantization.quant_err(grad_input)
        else:
            grad_input = grad_input

        return grad_input, None, None, None, None

class QLinearLayerSign(nn.Module):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    def __init__(self, input_features, output_features, pass_through = False, bias = True, dtype = None, device = None):
        super(QLinearLayerSign, self).__init__()
        self.input_features  = input_features
        self.output_features = output_features
        self.dtype = dtype
        self.device = device

        # weight and bias for forward pass
        self.weights   = nn.Parameter(torch.empty((output_features, input_features), device=device, dtype=dtype, requires_grad=False))
        self.weight_fa = nn.Parameter(torch.empty((output_features, input_features), device=device, dtype=dtype, requires_grad=False))
        self.bias      = nn.Parameter(torch.empty((output_features), device=device, dtype=dtype, requires_grad=False))

        if quantization.global_sb is not None:
            self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_sb)]))
            #self.L    = np.sqrt(6/self.input_features)
            self.L     = lc_ampl/np.sqrt(self.input_features)
            self.scale = 2 ** round(math.log(self.L_min / self.L, 2.0))
            self.scale = self.scale if self.scale > 1 else 1.0
            self.L     = np.max([self.L, self.L_min])

            #since those weights are fixed lets just initialize them between -1 and 1 to make use of all given bits 
            self.L     = lc_ampl/np.sqrt(self.input_features)
            self.scale = 2 ** round(math.log((1-self.L_min)/self.L, 2.0))
            self.scale = self.scale if self.scale > 1 else 1.0
            self.L     = 1

            torch.nn.init.uniform_(self.weights, a = -self.L, b = self.L)
            torch.nn.init.uniform_(self.weight_fa, a = -self.L, b = self.L)
            if bias:
                torch.nn.init.uniform_(self.bias, a = -self.L, b = self.L)
            else:
                self.bias = None

            # quantize them
            with torch.no_grad():
                self.weights.data   = quantization.quant_s(self.weights.data)
                self.weight_fa.data = quantization.quant_s(self.weight_fa.data)
                if self.bias is not None:
                    self.bias.data  = quantization.quant_s(self.bias.data)
        else:
            self.scale = 1
            self.stdv = lc_ampl/np.sqrt(self.input_features)
            torch.nn.init.uniform_(self.weights, a = -self.stdv, b = self.stdv)
            torch.nn.init.uniform_(self.weight_fa, a = -self.stdv, b = self.stdv)
            if bias:
                torch.nn.init.uniform_(self.bias, a = -self.stdv, b = self.stdv)
            else:
                self.bias = None

        # sign concordant weights in fwd and bwd pass
        #self.weight_fa = self.weights
        nonzero_mask = (self.weights.data != 0)
        self.weight_fa.data[nonzero_mask] *= torch.sign((torch.sign(self.weights.data) == torch.sign(self.weight_fa.data)).type(dtype) -.5)[nonzero_mask]

            
    def forward(self, input):
        return QLinearFunctional.apply(input, self.weights, self.weight_fa, self.bias, self.scale) 



class QSConv2dFunctional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, scale, padding = 0, weight_mult = 1):
        if quantization.global_wb is not None:
            w_quant = quantization.quant_w(weights/weight_mult, 1) *weight_mult
            bias_quant = quantization.quant_w(bias/weight_mult, 1) *weight_mult
        else:
            w_quant = weights
            bias_quant = bias
        ctx.padding = padding

        output = F.conv2d(input = input, weight = w_quant, bias = bias_quant, padding = ctx.padding)
        if quantization.global_wb is not None:
            output = output / scale

        ctx.save_for_backward(input, w_quant, bias_quant) 

        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w_quant, bias_quant = ctx.saved_tensors 
        grad_input = grad_weight = grad_bias = None 

        if quantization.global_eb is not None:
            quant_error = quantization.quant_err(grad_output)
        else:
            quant_error = grad_output

        # compute quantized error
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, w_quant, quant_error, padding = ctx.padding)
        # computed quantized gradient
        if ctx.needs_input_grad[1]:
            if quantization.global_gb is not None:
                grad_weight = quantization.quant_grad(torch.nn.grad.conv2d_weight(input, w_quant.shape, quant_error, padding = ctx.padding)).float()
            else:
                grad_weight = torch.nn.grad.conv2d_weight(input, w_quant.shape, quant_error, padding = ctx.padding)
        # computed quantized bias
        if bias_quant is not None and ctx.needs_input_grad[2]:
            if quantization.global_gb is not None:
                grad_bias = quantization.quant_grad(quant_error.sum((0,2,3)).squeeze(0)).float()
            else:
                grad_bias = quant_error.sum((0,2,3)).squeeze(0)

        if input.shape[2] == 13:
            quantization.global_w3update += grad_bias.nonzero().shape[0] + grad_weight.nonzero().shape[0] 
        if input.shape[2] == 15:
            quantization.global_w2update += grad_bias.nonzero().shape[0] + grad_weight.nonzero().shape[0] 
        if input.shape[2] == 32:
            quantization.global_w1update += grad_bias.nonzero().shape[0] + grad_weight.nonzero().shape[0] 
        return grad_input, grad_weight, grad_bias, None, None, None, None


class LIFConv2dLayer(nn.Module):
    def __init__(self, inp_shape, kernel_size, out_channels, tau_syn, tau_mem, tau_ref, delta_t, pooling = 1, padding = 0, bias = True, thr = 1, device=torch.device("cpu"), dtype = torch.float, dropout_p = .5, output_neurons = 10, loss_fn = None, l1 = 0, l2 = 0, PQ_cap = 1, weight_mult = 4e-5):
        super(LIFConv2dLayer, self).__init__()   
        self.device = device
        self.dtype = dtype
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.out_channels = out_channels         
        self.output_neurons = output_neurons
        self.padding = padding
        self.pooling = pooling
        self.thr = thr
        self.PQ_cap = PQ_cap
        self.weight_mult = weight_mult
        self.fan_in = kernel_size * kernel_size * inp_shape[0]
                
        self.dropout_learning = nn.Dropout(p=dropout_p)
        self.dropout_p = dropout_p
        self.l1 = l1
        self.l2 = l2
        self.loss_fn = loss_fn

        self.weights = nn.Parameter(torch.empty((self.out_channels, inp_shape[0],  self.kernel_size, self.kernel_size),  device=device, dtype=dtype, requires_grad=True))

        # decide which one you like
        self.stdv =  1 / np.sqrt(self.fan_in) #/ 250 * 1e-2
        #self.stdv =  np.sqrt(6 / self.fan_in) #* self.weight_mult
        if quantization.global_wb is not None:
            self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_wb)]))
            #self.stdv = np.sqrt(6/self.fan_in) 
            self.scale = 2 ** round(math.log(self.L_min / self.stdv, 2.0))
            self.scale = self.scale if self.scale > 1 else 1.0
            self.L     = np.max([self.stdv, self.L_min])
            torch.nn.init.uniform_(self.weights, a = -self.L * self.weight_mult, b = self.L* self.weight_mult)
        else:
            self.scale = 1
            torch.nn.init.uniform_(self.weights, a = -self.stdv * self.weight_mult, b = self.stdv* self.weight_mult)

        # bias has a different scale... just why?
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype, requires_grad=True))
            if quantization.global_wb is not None:
                bias_L = np.max([self.stdv* 1e2, self.L_min])
                torch.nn.init.uniform_(self.bias, a = -bias_L * self.weight_mult, b = bias_L* self.weight_mult)
            else:
                torch.nn.init.uniform_(self.bias, a = -self.stdv* self.weight_mult* 1e2, b = self.stdv* self.weight_mult * 1e2)
        else:
            self.register_parameter('bias', None)

        self.mpool = nn.MaxPool2d(kernel_size = self.pooling, stride = self.pooling, padding = (self.pooling-1)//2, return_indices=False)
        self.out_shape2 = self.mpool(QSConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape, dtype = dtype).to(device), self.weights, self.bias, self.scale, self.padding)).shape[1:] #self.pooling, 
        self.out_shape = QSConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape, dtype = dtype).to(device), self.weights, self.bias, self.scale, self.padding).shape[1:]
        
        self.sign_random_readout = QLinearLayerSign(input_features = np.prod(self.out_shape2), output_features = output_neurons, pass_through = False, bias = False, dtype = self.dtype, device = device).to(device)

        # tau quantization, static hardware friendly values
        if tau_syn.shape[0] == 2:
            self.tau_syn = torch.empty(torch.Size(self.inp_shape), dtype = dtype).uniform_(tau_syn[0], tau_syn[1]).to(device)
            self.beta    = 1. - 1e-3 / self.tau_syn
            self.tau_syn = 1. / (1. - self.beta)
        else:
            self.beta = torch.tensor([1 - delta_t / tau_syn], dtype = dtype).to(device) 
            self.tau_syn = 1. / (1. - self.beta)


        if tau_mem.shape[0] == 2:
            self.tau_mem = torch.empty(torch.Size(self.inp_shape), dtype = dtype).uniform_(tau_mem[0], tau_mem[1]).to(device)
            self.alpha   = 1. - 1e-3 / self.tau_mem
            self.tau_mem = 1. / (1. - self.alpha)
        else:
            self.alpha = torch.tensor([1 - delta_t / tau_mem], dtype = dtype).to(device) 
            self.tau_mem = 1. / (1. - self.alpha)


        if tau_ref.shape[0] == 2:
            self.tau_ref = torch.empty(torch.Size(self.inp_shape), dtype = dtype).uniform_(tau_ref[0], tau_ref[1]).to(device)
            self.gamma   = 1. - 1e-3 / self.tau_gamma
            self.tau_ref = 1. / (1. - self.gamma)
        else:
            self.gamma = torch.tensor([1 - delta_t / tau_ref], dtype = dtype).to(device)
            self.tau_ref = 1. / (1. - self.gamma)

        self.r_scale = 1/(1-self.gamma) # the one comes from decolle, best value ?
        #self.q_scale = self.tau_syn/(1-self.beta)
        #self.q_scale = self.q_scale.max()
        # p_scale should be max overall to differentiate input signals
        #self.p_scale = (self.tau_mem * self.q_scale*self.PQ_cap)/(1-self.alpha)
        #self.p_scale = self.p_scale.max()

        self.inp_mult_q = self.tau_syn##1/self.PQ_cap * (1-self.beta.max()) #
        self.inp_mult_p = self.tau_mem##1/self.PQ_cap * (1-self.alpha.max()) #
        #self.pmult = self.p_scale * self.PQ_cap * self.weight_mult

        # those might be clamped as in chop off values.
        self.Q_scale = (self.tau_syn/(1-self.beta)).max()
        self.P_scale = ((self.tau_mem * self.Q_scale)/(1-self.alpha)).max()
        self.Q_scale = (self.tau_syn/(1-self.beta)).max()
        self.R_scale = 1/(1-self.gamma)

        if quantization.global_wb is not None:
            with torch.no_grad():
                self.weights.data = quantization.quant_w(self.weights.data)
                if self.bias is not None:
                    self.bias.data = quantization.quant_w(self.bias.data)


    def state_init(self, batch_size):
        self.P = torch.zeros((batch_size,) + self.inp_shape, dtype = self.dtype).detach().to(self.device)
        self.Q = torch.zeros((batch_size,) + self.inp_shape, dtype = self.dtype).detach().to(self.device)
        self.R = torch.zeros((batch_size,) + self.out_shape, dtype = self.dtype).detach().to(self.device)
        self.S = torch.zeros((batch_size,) + self.out_shape, dtype = self.dtype).detach().to(self.device)
        self.U = torch.zeros((batch_size,) + self.out_shape, dtype = self.dtype).detach().to(self.device)

    
    def forward(self, input_t, y_local, train_flag = False, test_flag = False):
        # probably dont need to quantize because gb steps are arleady in the right level... just clipping
        if quantization.global_gb is not None:
            with torch.no_grad():
                self.weights.data = quantization.clip(self.weights.data/self.weight_mult, quantization.global_gb)*self.weight_mult
                if self.bias is not None:
                    self.bias.data = quantization.clip(self.bias.data/self.weight_mult, quantization.global_gb)*self.weight_mult
        if quantization.global_rfb is not None:
            # R always using full scale?
            self.R = quantization.quant01(self.R/self.R_scale, quantization.global_rfb)*self.R_scale

        #self.P, self.R, self.Q = self.alpha * self.P + self.tau_mem * self.Q, self.gamma * self.R, self.beta * self.Q + self.tau_syn * input_t
        #dtype necessary
        self.P, self.R, self.Q = self.alpha * self.P + self.inp_mult_p * self.Q, self.gamma * self.R, self.beta * self.Q + self.inp_mult_q * input_t.type(self.dtype)

        if self.PQ_cap != 1:
            self.P = torch.clamp(self.P, 0, self.P_scale*self.PQ_cap)
            self.Q = torch.clamp(self.Q, 0, self.Q_scale*self.PQ_cap)

        if quantization.global_pb is not None:
            self.P = torch.clamp(self.P/(self.P_scale*self.PQ_cap), 0, 1)
            self.P = quantization.quant01(self.P, quantization.global_pb)*(self.P_scale*self.PQ_cap)
        if quantization.global_qb is not None:
            self.Q = torch.clamp(self.Q/(self.Q_scale*self.PQ_cap), 0, 1)
            self.Q = quantization.quant01(self.Q, quantization.global_qb)*(self.Q_scale*self.PQ_cap)

        #self.U = QSConv2dFunctional.apply(self.P * self.pmult, self.weights, self.bias, self.scale, self.padding) - self.R
        self.U = QSConv2dFunctional.apply(self.P, self.weights, self.bias, self.scale, self.padding, self.weight_mult) - self.R #* self.r_scale 
        if quantization.global_ub is not None:
            self.U = quantU.apply(self.U)
        self.S = (self.U >= self.thr).type(self.dtype) #float()
        self.R += self.S * 1#(1-self.gamma)


        if test_flag or train_flag:
            self.U_aux = torch.sigmoid(self.U) # quantize this function.... at some point
            self.U_aux = self.mpool(self.U_aux)

            rreadout = self.dropout_learning(self.sign_random_readout(self.U_aux.reshape([input_t.shape[0], np.prod(self.out_shape2)]))) * self.dropout_p

            if train_flag:
                if quantization.global_eb is not None:
                    part1 = quantization.SSE(rreadout, y_local)
                    #part2 = self.l1 * 200e-1 * F.relu((self.U+.01)).mean()
                    part2 = self.l1 * 200e-1 * F.relu((self.U_aux+.01)).mean()
                    #part3 = self.l2 *1e-1* F.relu(.1-self.U_aux.mean())
                    part3 = self.l2 *1e-1* F.relu(.1-self.U.mean())
                    loss_gen = part1 + part2 + part3
                else:
                    part1 = self.loss_fn(rreadout, y_local)
                    #part2 = self.l1 * 200e-1 * F.relu((self.U+.01)).mean()
                    part2 = self.l1 * 200e-1 * F.relu((self.U_aux+.01)).mean()
                    #part3 = self.l2 *1e-1* F.relu(.1-self.U_aux.mean())
                    part3 = self.l2 *1e-1* F.relu(.1-self.U.mean())
                    loss_gen = part1 + part2 + part3
                    #loss_gen = self.loss_fn(rreadout, y_local) + self.l1 * 200e-1 * F.relu((self.U+.01)).mean() + self.l2 *1e-1* F.relu(.1-self.U_aux.mean())
                #loss_gen = self.loss_fn(rreadout, y_local) + self.l1 * 200e-1 * F.relu((self.U+.01)).mean() + self.l2 *1e-1* F.relu(.1-self.U_aux.mean())
            else:
                part1 = None
                part2 = None
                part3 = None
                loss_gen = None
        else:
            part1 = None
            part2 = None
            part3 = None
            loss_gen = None
            rreadout = torch.tensor([[0]])


        return self.mpool(self.S), loss_gen, rreadout.argmax(1), [part1, part2, part3]



class DTNLIFConv2dLayer(nn.Module):
    def __init__(self, inp_shape, kernel_size, out_channels, tau_syn, tau_mem, tau_ref, delta_t, pooling = 1, padding = 0, bias = True, thr = 1, device=torch.device("cpu"), dtype = torch.float, dropout_p = .5, output_neurons = 10, loss_fn = None, l1 = 0, l2 = 0, PQ_cap = 1, weight_mult = 4e-5):
        super(DTNLIFConv2dLayer, self).__init__()   
        self.device = device
        self.dtype = dtype
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.out_channels = out_channels         
        self.output_neurons = output_neurons
        self.padding = padding
        self.pooling = pooling
        self.thr = thr
        self.PQ_cap = PQ_cap
        self.weight_mult = weight_mult
        self.fan_in = kernel_size * kernel_size * inp_shape[0]
                
        self.dropout_learning = nn.Dropout(p=dropout_p)
        self.dropout_p = dropout_p
        self.l1 = l1
        self.l2 = l2
        self.loss_fn = loss_fn

        self.weights = nn.Parameter(torch.empty((self.out_channels, inp_shape[0],  self.kernel_size, self.kernel_size),  device=device, dtype=dtype, requires_grad=True))

        # decide which one you like
        self.stdv =  1 / np.sqrt(self.fan_in) #/ 250 * 1e-2
        #self.stdv =  np.sqrt(6 / self.fan_in) #* self.weight_mult
        if quantization.global_wb is not None:
            self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_wb)]))
            #self.stdv = np.sqrt(6/self.fan_in) 
            self.scale = 2 ** round(math.log(self.L_min / self.stdv, 2.0))
            self.scale = self.scale if self.scale > 1 else 1.0
            self.L     = np.max([self.stdv, self.L_min])
            torch.nn.init.uniform_(self.weights, a = -self.L * self.weight_mult, b = self.L* self.weight_mult)
        else:
            self.scale = 1
            torch.nn.init.uniform_(self.weights, a = -self.stdv * self.weight_mult, b = self.stdv* self.weight_mult)

        # bias has a different scale... just why?
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype, requires_grad=True))
            if quantization.global_wb is not None:
                bias_L = np.max([self.stdv* 1e2, self.L_min])
                torch.nn.init.uniform_(self.bias, a = -bias_L * self.weight_mult, b = bias_L* self.weight_mult)
            else:
                torch.nn.init.uniform_(self.bias, a = -self.stdv* self.weight_mult* 1e2, b = self.stdv* self.weight_mult * 1e2)
        else:
            self.register_parameter('bias', None)

        self.mpool = nn.MaxPool2d(kernel_size = self.pooling, stride = self.pooling, padding = (self.pooling-1)//2, return_indices=False)
        self.out_shape2 = self.mpool(QSConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape, dtype = dtype).to(device), self.weights, self.bias, self.scale, self.padding)).shape[1:] #self.pooling, 
        self.out_shape = QSConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape, dtype = dtype).to(device), self.weights, self.bias, self.scale, self.padding).shape[1:]
        
        self.sign_random_readout = QLinearLayerSign(input_features = np.prod(self.out_shape2), output_features = output_neurons, pass_through = False, bias = False, dtype = self.dtype, device = device).to(device)

        # tau quantization, static hardware friendly values
        if tau_syn.shape[0] == 2:
            self.tau_syn = torch.empty(torch.Size(self.inp_shape), dtype = dtype).uniform_(tau_syn[0], tau_syn[1]).to(device)
            self.beta    = 1. - 1e-3 / self.tau_syn
            self.tau_syn = 1. / (1. - self.beta)
        else:
            self.beta = torch.tensor([1 - delta_t / tau_syn], dtype = dtype).to(device) 
            self.tau_syn = 1. / (1. - self.beta)


        if tau_mem.shape[0] == 2:
            self.tau_mem = torch.empty(torch.Size(self.inp_shape), dtype = dtype).uniform_(tau_mem[0], tau_mem[1]).to(device)
            self.alpha   = 1. - 1e-3 / self.tau_mem
            self.tau_mem = 1. / (1. - self.alpha)
        else:
            self.alpha = torch.tensor([1 - delta_t / tau_mem], dtype = dtype).to(device) 
            self.tau_mem = 1. / (1. - self.alpha)


        if tau_ref.shape[0] == 2:
            self.tau_ref = torch.empty(torch.Size(self.inp_shape), dtype = dtype).uniform_(tau_ref[0], tau_ref[1]).to(device)
            self.gamma   = 1. - 1e-3 / self.tau_gamma
            self.tau_ref = 1. / (1. - self.gamma)
        else:
            self.gamma = torch.tensor([1 - delta_t / tau_ref], dtype = dtype).to(device)
            self.tau_ref = 1. / (1. - self.gamma)

        self.r_scale = 1/(1-self.gamma) # the one comes from decolle, best value ?
        #self.q_scale = self.tau_syn/(1-self.beta)
        #self.q_scale = self.q_scale.max()
        # p_scale should be max overall to differentiate input signals
        #self.p_scale = (self.tau_mem * self.q_scale*self.PQ_cap)/(1-self.alpha)
        #self.p_scale = self.p_scale.max()

        self.inp_mult_q = self.tau_syn##1/self.PQ_cap * (1-self.beta.max()) #
        self.inp_mult_p = self.tau_mem##1/self.PQ_cap * (1-self.alpha.max()) #
        #self.pmult = self.p_scale * self.PQ_cap * self.weight_mult

        # those might be clamped as in chop off values.
        self.Q_scale = (self.tau_syn/(1-self.beta)).max()
        self.P_scale = ((self.tau_mem * self.Q_scale)/(1-self.alpha)).max()
        self.Q_scale = (self.tau_syn/(1-self.beta)).max()
        self.R_scale = 1/(1-self.gamma)

        if quantization.global_wb is not None:
            with torch.no_grad():
                self.weights.data = quantization.quant_w(self.weights.data)
                if self.bias is not None:
                    self.bias.data = quantization.quant_w(self.bias.data)


    def state_init(self, batch_size):
        self.P = torch.zeros((batch_size,) + self.inp_shape, dtype = self.dtype).detach().to(self.device)
        self.Q = torch.zeros((batch_size,) + self.inp_shape, dtype = self.dtype).detach().to(self.device)
        self.R = torch.zeros((batch_size,) + self.out_shape, dtype = self.dtype).detach().to(self.device)
        self.S = torch.zeros((batch_size,) + self.out_shape, dtype = self.dtype).detach().to(self.device)
        self.U = torch.zeros((batch_size,) + self.out_shape, dtype = self.dtype).detach().to(self.device)

    
    def forward(self, input_t, y_local, train_flag = False, test_flag = False):
        # probably dont need to quantize because gb steps are arleady in the right level... just clipping
        if quantization.global_gb is not None:
            with torch.no_grad():
                self.weights.data = quantization.clip(self.weights.data/self.weight_mult, quantization.global_gb)*self.weight_mult
                if self.bias is not None:
                    self.bias.data = quantization.clip(self.bias.data/self.weight_mult, quantization.global_gb)*self.weight_mult
        if quantization.global_rfb is not None:
            # R always using full scale?
            self.R = quantization.quant01(self.R/self.R_scale, quantization.global_rfb)*self.R_scale

        #self.P, self.R, self.Q = self.alpha * self.P + self.tau_mem * self.Q, self.gamma * self.R, self.beta * self.Q + self.tau_syn * input_t
        #dtype necessary
        self.P, self.R, self.Q = self.alpha * self.P + self.inp_mult_p * self.Q, self.gamma * self.R, self.beta * self.Q + self.inp_mult_q * input_t.type(self.dtype)

        if self.PQ_cap != 1:
            self.P = torch.clamp(self.P, 0, self.P_scale*self.PQ_cap)
            self.Q = torch.clamp(self.Q, 0, self.Q_scale*self.PQ_cap)

        if quantization.global_pb is not None:
            self.P = torch.clamp(self.P/(self.P_scale*self.PQ_cap), 0, 1)
            self.P = quantization.quant01(self.P, quantization.global_pb)*(self.P_scale*self.PQ_cap)
        if quantization.global_qb is not None:
            self.Q = torch.clamp(self.Q/(self.Q_scale*self.PQ_cap), 0, 1)
            self.Q = quantization.quant01(self.Q, quantization.global_qb)*(self.Q_scale*self.PQ_cap)

        #self.U = QSConv2dFunctional.apply(self.P * self.pmult, self.weights, self.bias, self.scale, self.padding) - self.R
        self.U = QSConv2dFunctional.apply(self.P, self.weights, self.bias, self.scale, self.padding, self.weight_mult) - self.R #* self.r_scale 
        if quantization.global_ub is not None:
            self.U = quantU.apply(self.U)
        self.S = (self.U >= self.thr).type(self.dtype)
        self.S += (self.U <= -self.thr).type(self.dtype)*-1
        self.R += self.S * self.thr#(1-self.gamma)


        if test_flag or train_flag:
            self.U_aux = torch.sigmoid(self.U) # quantize this function.... at some point
            self.U_aux = self.mpool(self.U_aux)

            rreadout = self.dropout_learning(self.sign_random_readout(self.U_aux.reshape([input_t.shape[0], np.prod(self.out_shape2)]))) * self.dropout_p

            if train_flag:
                if quantization.global_eb is not None:
                    part1 = quantization.SSE(rreadout, y_local)
                    part2 = self.l1 * 200e-1 * F.relu((self.U+.01)).mean()
                    part3 = self.l2 *1e-1* F.relu(.1-self.U_aux.mean())
                    loss_gen = part1 + part2 + part3
                else:
                    part1 = self.loss_fn(rreadout, y_local)
                    part2 = self.l1 * 200e-1 * F.relu((self.U+.01)).mean()
                    part3 = self.l2 *1e-1* F.relu(.1-self.U_aux.mean())
                    loss_gen = part1 + part2 + part3
                #loss_gen = self.loss_fn(rreadout, y_local) + self.l1 * 200e-1 * F.relu((self.U+.01)).mean() + self.l2 *1e-1* F.relu(.1-self.U_aux.mean())
            else:
                loss_gen = None
        else:
            loss_gen = None
            rreadout = torch.tensor([[0]])


        return self.mpool(self.S), loss_gen, rreadout.argmax(1), [part1, part2, part3]


