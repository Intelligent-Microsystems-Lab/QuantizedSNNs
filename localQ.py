import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import quantization

lc_ampl = .5

def create_graph(plot_file_name, diff_layers_acc):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8.4, 4.8)
    plt.title("DVS Gesture" + " B" + str(quantization.global_ab) + " LRB" + str(quantization.global_sb))
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

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(t, diff_layers_acc['loss'], 'k-', label = 'Loss')

    fig.tight_layout()
    plt.savefig(plot_file_name)
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
    #bi directional two channels
    if input_mode == 3:
        down_spikes = torch.cat((x_local, x_local), dim = 1)
        down_spikes[down_spikes != 0] = 1
        return down_spikes
    else:
        print("No valid input mode")
        return -1

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
            yield sparse_matrix.to(device=device), y_batch.to(device=device)
            counter += 1
        except StopIteration:
            return

def sparse_data_generator_DVSGesture(X, y, batch_size, nb_steps, shuffle, device, test = False):
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
        #sparse_matrix[sparse_matrix < 0] = -1
        #sparse_matrix[sparse_matrix > 0] = 1

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

class QLinearFunctional(torch.autograd.Function):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    @staticmethod
    def forward(ctx, input, weight, weight_fa, bias=None):
        output = torch.einsum('ab,cb->ac', input, weight)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # quant act
        if quantization.global_ab is not None:
            output, clip_info = quantization.quant_act(output)
        else:
            clip_info = None

        ctx.save_for_backward(input, weight, weight_fa, bias, clip_info)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_fa, bias, clip_info = ctx.saved_tensors
        grad_input = None

        if quantization.global_eb is not None:
            quant_error = quantization.quant_err(grad_output) * clip_info.float()
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
    # we dont have a bias 
    def __init__(self, input_features, output_features, pass_through = False, bias = True):
        super(QLinearLayerSign, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # weight and bias for forward pass
        self.weights = nn.Parameter(torch.Tensor(output_features, input_features), requires_grad=False)
        self.weight_fa = nn.Parameter(torch.Tensor(output_features, input_features), requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(output_features), requires_grad=False)

        if quantization.global_sb is not None:
            self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_sb)]))
            #self.L = np.max([np.sqrt(6/self.input_features), self.L_min])
            self.L = np.max([lc_ampl/np.sqrt(torch.tensor(self.weights.shape).prod().item()), self.L_min])
            self.scale = 2 ** round(math.log(self.L_min / self.L, 2.0))
            self.scale = self.scale if self.scale > 1 else 1.0


            torch.nn.init.uniform_(self.weights, a = -self.L, b = self.L)
            torch.nn.init.uniform_(self.weight_fa, a = -self.L, b = self.L)
            if bias:
                torch.nn.init.uniform_(self.bias, a = -self.L, b = self.L)
            else:
                self.bias = None

            # quantize them
            with torch.no_grad():
                self.weights.data = quantization.quant_w_custom(self.weights.data, quantization.global_sb, self.scale)
                self.weight_fa.data = quantization.quant_w_custom(self.weight_fa.data, quantization.global_sb, self.scale)
                if self.bias is not None:
                    self.bias.data = quantization.quant_w_custom(self.bias.data, quantization.global_sb, self.scale)
        else:
            self.stdv = lc_ampl/np.sqrt(torch.tensor(self.weights.shape).prod().item())
            torch.nn.init.uniform_(self.weights, a = -self.stdv, b = self.stdv)
            torch.nn.init.uniform_(self.weight_fa, a = -self.stdv, b = self.stdv)
            if bias:
                torch.nn.init.uniform_(self.bias, a = -self.stdv, b = self.stdv)
            else:
                self.bias = None

        # sign concordant weights in fwd and bwd pass
        #self.weight_fa = self.weights
        self.weight_fa.data *= torch.sign((torch.sign(self.weights.data) == torch.sign(self.weight_fa.data)).float() -.5)
            
        
    def forward(self, input):
        return QLinearFunctional.apply(input, self.weights, self.weight_fa, self.bias)



class QSConv2dFunctional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, scale, padding = 0):
        if quantization.global_wb is not None:
            w_quant = quantization.quant_w(weights, scale)
            bias_quant = quantization.quant_w(bias, scale)
        else:
            w_quant = weights
            bias_quant = bias
        ctx.padding = padding

        output = F.conv2d(input = input, weight = w_quant, bias = bias_quant, padding = ctx.padding)

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

        return grad_input, grad_weight, grad_bias, None, None, None, None


class LIFConv2dLayer(nn.Module):
    def __init__(self, inp_shape, kernel_size, out_channels, tau_syn, tau_mem, tau_ref, delta_t, pooling = 1, padding = 0, bias = True, thr = 1, device=torch.device("cpu"), dtype = torch.float, dropout_p = .5, output_neurons = 10, loss_fn = None, l1 = 0, l2 = 0):
        super(LIFConv2dLayer, self).__init__()   
        self.device = device
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.out_channels = out_channels         
        self.output_neurons = output_neurons
        self.padding = padding
        self.pooling = pooling
        self.thr = thr
                
        self.dropout_learning = nn.Dropout(p=dropout_p)
        self.dropout_p = dropout_p
        self.l1 = l1
        self.l2 = l2
        self.loss_fn = loss_fn

        self.weights = nn.Parameter(torch.empty((self.out_channels, inp_shape[0],  self.kernel_size, self.kernel_size),  device=device, dtype=dtype, requires_grad=True))

        self.stdv =  1 / np.sqrt(torch.tensor(self.weights.shape).prod().item()) / 250
        if (quantization.global_gb is not None) or (quantization.global_wb is not None): 
            self.fan_in = kernel_size * kernel_size * inp_shape[0]
            self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_wb)]))
            self.L = np.max([1 / np.sqrt(torch.tensor(self.weights.shape).prod().item()) / 250 *1e-2, self.L_min])
            #self.L = np.max([np.sqrt( 6/self.fan_in), self.L_min])
            self.scale = 2 ** round(math.log(self.L_min / self.L, 2.0))
            self.scale = self.scale if self.scale > 1 else 1.0
            torch.nn.init.uniform_(self.weights, a = -self.L, b = self.L)
        else:
            torch.nn.init.uniform_(self.weights, a = -self.stdv*1e-2, b = self.stdv*1e-2)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype, requires_grad=True))
            if (quantization.global_gb is not None) or (quantization.global_wb is not None):
                self.L_bias = np.max([1 / np.sqrt(torch.tensor(self.weights.shape).prod().item()) / 250, self.L_min]) 
                torch.nn.init.uniform_(self.bias, a = -self.L_bias, b = self.L_bias)
            else:
                torch.nn.init.uniform_(self.bias, a = -self.stdv, b = self.stdv)
        else:
            self.register_parameter('bias', None)

        self.mpool = nn.MaxPool2d(kernel_size = self.pooling, stride = self.pooling, padding = (self.pooling-1)//2, return_indices=False)
        self.out_shape2 = self.mpool(QSConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape).to(device), self.weights, self.bias, self.scale, self.padding)).shape[1:] #self.pooling, 
        self.out_shape = QSConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape).to(device), self.weights, self.bias, self.scale, self.padding).shape[1:]
        
        self.sign_random_readout = QLinearLayerSign(input_features = np.prod(self.out_shape2), output_features = output_neurons, pass_through = False, bias = False).to(device)

        # tau quantization.....
        if tau_syn.shape[0] == 2:
            self.tau_syn = torch.Tensor(torch.Size(self.inp_shape)).uniform_(tau_syn[0], tau_syn[1]).to(device)
            self.beta    = 1. - 1e-3 / self.tau_syn
            self.tau_syn = 1. / (1. - self.beta)
        else:
            self.beta = torch.Tensor([torch.exp( - delta_t / tau_syn)]).to(device)


        if tau_mem.shape[0] == 2:
            self.tau_mem = torch.Tensor(torch.Size(self.inp_shape)).uniform_(tau_mem[0], tau_mem[1]).to(device)
            self.alpha   = 1. - 1e-3 / self.tau_mem
            self.tau_mem = 1. / (1. - self.alpha)
        else:
            self.alpha = torch.Tensor([torch.exp( - delta_t / tau_mem)]).to(device)


        if tau_ref.shape[0] == 2:
            self.gamma = torch.exp( -delta_t / torch.Tensor(torch.Size(self.out_shape)).uniform_(tau_ref[0], tau_ref[1]).to(device))
        else:
            self.gamma = torch.Tensor([torch.exp( - delta_t / tau_ref)]).to(device)


        if (quantization.global_gb is not None) or (quantization.global_wb is not None):
            with torch.no_grad():
                self.weights.data = quantization.quant_w_custom(self.weights.data, quantization.global_wb, self.scale)
                if self.bias is not None:
                    self.bias.data = quantization.quant_w_custom(self.bias.data, quantization.global_wb, self.scale)

        self.L_minw = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_wb)]))
        self.Lw = np.max([1 / np.sqrt(torch.tensor(self.weights.shape).prod().item()) / 250 *1e-2, self.L_min])
        #self.L = np.max([np.sqrt( 6/self.fan_in), self.L_min])
        self.scalew = 2 ** round(math.log(self.L_min / self.L, 2.0))
        self.scalew = self.scale if self.scale > 1 else 1.0

    def state_init(self, batch_size):
        self.P = torch.zeros((batch_size,) + self.inp_shape).detach().to(self.device)
        self.Q = torch.zeros((batch_size,) + self.inp_shape).detach().to(self.device)
        self.R = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)
        self.S = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)
        self.U = torch.zeros((batch_size,) + self.out_shape).detach().to(self.device)

    
    def forward(self, input_t, y_local, train_flag = False, test_flag = False):
        if quantization.global_gb is not None:
            with torch.no_grad():
                self.weights.data = quantization.quant_w_custom(self.weights.data, quantization.global_gb, 1)
                if self.bias is not None:
                    self.bias.data = quantization.quant_w_custom(self.bias.data, quantization.global_gb, 1)

        self.P, self.R, self.Q = self.alpha * self.P + self.tau_mem * self.Q, 0.65 * self.R, self.beta * self.Q + self.tau_syn * input_t

        # quantize P, Q
        if quantization.global_pb is not None:
            self.P = torch.clamp(torch.round(self.P), -quantization.step_d(quantization.global_pb)+1, quantization.step_d(quantization.global_pb))
        if quantization.global_qb is not None:
            self.Q = torch.clamp(torch.round(self.Q), -quantization.step_d(quantization.global_qb)+1, quantization.step_d(quantization.global_qb))
            #self.P, _ = quantization.quant_generic(self.P, quantization.global_pb)
            #self.Q, _ = quantization.quant_generic(self.Q, quantization.global_qb)

        self.U = QSConv2dFunctional.apply(self.P, self.weights, self.bias, self.scalew, self.padding) + self.R 

        # quantize U
        if quantization.global_ub is not None:
            self.U, _ = quantization.quant_generic(self.U, quantization.global_ub)

        self.S = (self.U >= self.thr).float()
        self.R -= self.S * 1

        if quantization.global_rb is not None:
            self.R, _ = quantization.quant_generic(self.R, quantization.global_rb)

        if test_flag or train_flag:
            self.U_aux = torch.sigmoid(self.U) # quantize this function.... at some point
            self.U_aux = self.mpool(self.U_aux)

            rreadout = self.dropout_learning(self.sign_random_readout(self.U_aux.reshape([input_t.shape[0], np.prod(self.out_shape2)]) ))

            if train_flag:
                import pdb; pdb.set_trace()
                if quantization.global_eb is not None:
                    loss_gen = self.loss_fn(rreadout, y_local) + self.l1 * 200e-1 * F.relu((self.U+.01)).mean() + self.l2 *1e-1* F.relu(.1-self.U_aux.mean())
                else:
                    loss_gen = self.loss_fn(rreadout, y_local) + self.l1 * 200e-1 * F.relu((self.U+.01)).mean() + self.l2 *1e-1* F.relu(.1-self.U_aux.mean())
                #loss_gen = self.loss_fn(rreadout, y_local) + self.l1 * 200e-1 * F.relu((self.U+.01)).mean() + self.l2 *1e-1* F.relu(.1-self.U_aux.mean())
            else:
                loss_gen = None
        else:
            loss_gen = None
            rreadout = torch.tensor([[0]])


        return self.mpool(self.S), loss_gen, rreadout.argmax(1)



