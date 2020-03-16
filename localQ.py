import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
import time
import math
import numpy as np

import quantization


def clee_spikes(T, rates):
    spikes = np.ones((T, + np.prod(rates.shape)))        
    spikes[np.random.binomial(1, (1000. - rates.flatten())/1000, size=(T, np.prod(rates.shape))).astype('bool')] = 0
    return spikes.T.reshape((rates.shape + (T,)))

def prep_input(x_local, input_mode):
    #two channel trick
    if input_mode == 0:
        down_spikes = torch.cat((x_local, x_local), dim = 1)
        mask1 = (down_spikes > 0) # this might change
        mask2 = (down_spikes < 0)
        mask1[:,0,:,:] = False
        mask2[:,1,:,:] = False
        down_spikes = torch.zeros_like(down_spikes)
        down_spikes[mask1] = 1 
        down_spikes[mask2] = 1
        return down_spikes
    # same same but different
    if input_mode == 1:
        down_spikes = x_local
        down_spikes[down_spikes != 0] = 1
        return down_spikes
    #bi directional
    if input_mode == 2:
        return x_local
    else:
        print("No valid input mode")
        return -1

def sparse_data_generator_DVSPoker(X, y, batch_size, nb_steps, shuffle, device, test = False):
    number_of_batches = len(y)//batch_size
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
            yield sparse_matrix.to(device=device), y_batch.to(device=device)
            counter += 1
        except StopIteration:
            return

def sparse_data_generator_DVSGesture(X, y, batch_size, nb_steps, shuffle, device, test = False):
    number_of_batches = len(y)//batch_size
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

        for bc,idx in enumerate(batch_index):
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
        all_events[:,4][all_events[:,4] == 0] = -1
        all_events = all_events[:,[0,2,3,1,4]]
        sparse_matrix = torch.sparse.FloatTensor(torch.LongTensor(all_events[:,[True, True, True, True, False]].T), torch.FloatTensor(all_events[:,4])).to_dense()

        # quick trick...
        sparse_matrix[sparse_matrix < 0] = -1
        sparse_matrix[sparse_matrix > 0] = 1

        sparse_matrix = sparse_matrix.reshape(torch.Size([sparse_matrix.shape[0], 1, sparse_matrix.shape[1], sparse_matrix.shape[2], sparse_matrix.shape[3]]))

        y_batch = torch.tensor(y[batch_index], dtype = int)
        try:
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
            yield X_batch.to(device), y_batch.to(device)
            counter += 1
        except StopIteration:
            return

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''
    @staticmethod
    def forward(ctx, x, quant_on = True):
        ctx.quant_on = quant_on
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):     
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input[x <= -.5] = 0
        grad_input[x > .5] = 0

        # quantize error
        if ctx.quant_on:
            grad_input = quantization.quant_err(grad_input)

        return grad_input, None

smoothstep = SmoothStep().apply

class QLinearFunctional(torch.autograd.Function):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    @staticmethod
    def forward(ctx, input, weight, bias=None, quant_on = True):
        ctx.quant_on = quant_on
        input[input > 0]  = 1 #correct for dropout scale
        input[input <= 0] = 0
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # quantize error - this function should receive a quant error***
        # 2 bits (-1, 1) * 1 bit (0, 1/spikes)
        if ctx.needs_input_grad[0]:
            if ctx.quant_on:
                grad_input = quantization.quant_err(grad_output.mm(weight))
            else:
                grad_input = grad_output.mm(weight)

        # those weights should not be updated
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.zeros_like(bias)

        return grad_input, grad_weight, grad_bias, None

class QLinearLayerSign(nn.Module):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    def __init__(self, input_features, output_features, pass_through = False, quant_on = True):
        super(QLinearLayerSign, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.quant_on = quant_on

        # weight and bias for forward pass
        self.weights = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weights.data.uniform_(-1, 1)

        if pass_through:
            self.weights.data = torch.ones_like(self.weights.data)

        if quant_on and not pass_through:
            scale = quantization.step_d(quantization.global_sb)
            s_sign = torch.sign(self.weights.data)
            self.weights.data = torch.ceil(torch.abs(self.weights.data) * scale ) / scale * s_sign
        
    def forward(self, input):
        return QLinearFunctional.apply(input, self.weights, None, self.quant_on)



class QSConv2dFunctional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, scale, padding = 0, pooling = None, quant_on = True):
        if quant_on:
            w_quant = quantization.quant_w(weights, scale)
            bias_quant = quantization.quant_w(bias, scale)
        else:
            w_quant = weights
            bias_quant = bias
        ctx.padding = padding
        ctx.pooling = pooling 
        ctx.size_pool = None
        ctx.quant_on = quant_on
        pool_indices = torch.ones(0)

        output = F.conv2d(input = input, weight = w_quant, bias = bias_quant, padding = ctx.padding)
        
        if ctx.pooling is not None:
            mpool = nn.MaxPool2d(kernel_size = ctx.pooling, stride = ctx.pooling, padding = (ctx.pooling-1)//2, return_indices=True)
            ctx.size_pool = output.shape
            output, pool_indices = mpool(output)

        ctx.save_for_backward(input, w_quant, bias_quant, pool_indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w_quant, bias_quant, pool_indices = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None 
        unmpool = nn.MaxUnpool2d(ctx.pooling, stride = ctx.pooling, padding = (ctx.pooling-1)//2)

        if ctx.pooling is not None:
            grad_output = unmpool(grad_output, pool_indices, output_size = torch.Size(ctx.size_pool))

        if ctx.quant_on:
            quant_error = quantization.quant_err(grad_output)
        else:
            quant_error = grad_output

        # compute quantized error
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, w_quant, quant_error, padding = ctx.padding)
        # computed quantized gradient
        if ctx.needs_input_grad[1]:
            if ctx.quant_on:
                grad_weight = quantization.quant_grad(torch.nn.grad.conv2d_weight(input, w_quant.shape, quant_error, padding = ctx.padding)).float()
            else:
                grad_weight = torch.nn.grad.conv2d_weight(input, w_quant.shape, quant_error, padding = ctx.padding)
        # computed quantized bias
        if bias_quant is not None and ctx.needs_input_grad[2]:
            if ctx.quant_on:
                grad_bias = quantization.quant_grad(torch.einsum("abcd->b",(quant_error))).float()
            else:
                grad_bias = torch.einsum("abcd->b",(quant_error))

        return grad_input, grad_weight, grad_bias, None, None, None, None


class LIFConv2dLayer(nn.Module):
    def __init__(self, inp_shape, kernel_size, out_channels, tau_syn, tau_mem, tau_ref, delta_t, pooling = 1, padding = 0, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float, dropout_p = .5, output_neurons = 10, loss_prep_fn = None, loss_fn = None, l1 = 0, l2 = 0, quant_on = False):
        super(LIFConv2dLayer, self).__init__()   
        self.device = device
        self.quant_on = quant_on
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.out_channels = out_channels  
        self.fan_in = kernel_size * kernel_size * inp_shape[0]
        self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_wb)]))
        self.L = np.max([np.sqrt( 6/self.fan_in), self.L_min])
        self.scale = 2 ** round(math.log(self.L_min / self.L, 2.0))
        self.scale = self.scale if self.scale > 1 else 1.0
        self.output_neurons = output_neurons

        self.padding = padding
        self.pooling = pooling
                
        self.dropout_learning = nn.Dropout(p=dropout_p)
        self.dropout_p = dropout_p
        self.l1 = l1
        self.l2 = l2
        self.loss_fn = loss_fn
        self.loss_prep_fn = loss_prep_fn

        self.weights = nn.Parameter(torch.empty((self.out_channels, inp_shape[0],  self.kernel_size, self.kernel_size),  device=device, dtype=dtype, requires_grad=True))
        if self.quant_on:
            torch.nn.init.uniform_(self.weights, a = -self.L, b = self.L)
        else:
            torch.nn.init.uniform_(self.weights, a = -1/torch.tensor(self.weights.shape).prod().item(), b = 1/torch.tensor(self.weights.shape).prod().item())


        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype, requires_grad=True))
            if self.quant_on:
                torch.nn.init.uniform_(self.bias, a = -self.L, b = self.L)
            else:
                torch.nn.init.uniform_(self.bias, a = -.04/torch.tensor(self.weights.shape).prod().item(), b = .04/torch.tensor(self.weights.shape).prod().item())
        else:
            self.register_parameter('bias', None)

        self.out_shape = QSConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape).to(device), self.weights, self.bias, self.scale, self.padding, self.pooling, self.quant_on).shape[1:]
        self.thr = thr

        self.sign_random_readout = QLinearLayerSign(np.prod(self.out_shape), output_neurons, self.quant_on).to(device)

        if tau_syn.shape[0] == 2:
            self.beta = torch.exp( -delta_t / torch.Tensor(torch.Size(self.inp_shape)).uniform_(tau_syn[0], tau_syn[1]).to(device))
        else:
            self.beta = torch.Tensor([torch.exp( - delta_t / tau_syn)]).to(device)
        if tau_mem.shape[0] == 2:
            #2, 32, 32
            self.alpha = torch.exp( -delta_t / torch.Tensor(torch.Size(self.inp_shape)).uniform_(tau_mem[0], tau_mem[1]).to(device))
        else:
            self.alpha = torch.Tensor([torch.exp( - delta_t / tau_mem)]).to(device)

        if tau_ref.shape[0] == 2:
            self.gamma = torch.exp( -delta_t / torch.Tensor(torch.Size(self.out_shape)).uniform_(tau_ref[0], tau_ref[1]).to(device))
        else:
            self.gamma = torch.Tensor([torch.exp( - delta_t / tau_ref)]).to(device)

        if self.quant_on:
            with torch.no_grad():
                self.weights.data = quantization.clip(quantization.quant_generic(self.weights.data, quantization.global_gb)[0], quantization.global_wb)
                if self.bias is not None:
                    self.bias.data = quantization.clip(quantization.quant_generic(self.bias.data, quantization.global_gb)[0], quantization.global_wb)

    def state_init(self, batch_size):
        self.P = torch.zeros((batch_size,) + self.inp_shape).detach().to(self.device)
        self.Q = torch.zeros((batch_size,) + self.inp_shape).detach().to(self.device)
        self.R = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)
        self.S = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)
        self.U = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)

    
    def forward(self, input_t, y_local):
        if self.quant_on:
            with torch.no_grad():
                self.weights.data = quantization.clip(self.weights.data, quantization.global_gb)
                if self.bias is not None:
                    self.bias.data = quantization.clip(self.bias.data, quantization.global_gb)

        # R could be used for refrac... right now its doing nothing....
        self.P, self.R, self.Q = 0.97 * self.P + self.Q, 0.65 * self.R + self.S, 0.92 * self.Q + input_t
        #self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R + self.S, self.beta * self.Q + input_t

        # quantize P, Q
        if self.quant_on:
            self.P, _ = quantization.quant_generic(self.P, quantization.global_pb)
            self.Q, _ = quantization.quant_generic(self.Q, quantization.global_qb)

        self.U = QSConv2dFunctional.apply(self.P, self.weights, self.bias, self.scale, self.padding, self.pooling, self.quant_on) - self.R

        # quantize U
        if self.quant_on:
            self.U, _ = quantization.quant_generic(self.U, quantization.global_ub)

        self.S = (self.U >= self.thr).float()
        # reset neurons which spiked
        #self.U = self.U * (1-self.S)

        rreadout = self.sign_random_readout(self.dropout_learning(smoothstep(self.U-self.thr, self.quant_on).reshape([input_t.shape[0], np.prod(self.out_shape)])) * self.dropout_p)
        _, predicted = torch.max(rreadout.data, 1)

        if y_local.shape[1] == self.output_neurons:
            correct_train = (predicted == y_local.max(dim = 1)[1]).sum().item()
        else:
            correct_train = (predicted == y_local).sum().item()

        loss_gen = self.loss_fn(self.loss_prep_fn(rreadout), y_local) + self.l1 * F.relu(self.U+.01).mean() + self.l2 * F.relu(self.thr-self.U.mean())
        #loss_gen = self.loss_fn(self.loss_prep_fn(rreadout), y_local) + self.l1 * F.relu((self.U+.01).mean()) + self.l2 * F.relu(self.thr-self.U).mean()
        #loss_gen = self.loss_fn(self.loss_prep_fn(rreadout), y_local) + self.l1 * F.relu((self.U+.01).mean()) + self.l2 * F.relu(self.thr-self.U.mean())
        #loss_gen = self.loss_fn(self.loss_prep_fn(rreadout), y_local) + self.l1 * F.relu(self.U+.01).mean() + self.l2 * F.relu(self.thr-self.U).mean()

        return self.S, loss_gen, correct_train


