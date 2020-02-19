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

def sparse_data_generator(X, y, batch_size, nb_steps, samples, max_hertz, shuffle=True, device=torch.device("cpu")):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors. 

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """
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
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):     
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input[x <= -.5] = 0
        grad_input[x > .5] = 0

        # quantize error
        grad_input = quantization.quant_err(grad_input)

        return grad_input

smoothstep = SmoothStep().apply


class SuperSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, x):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input = grad_input/(SuperSpike.scale*torch.abs(x)+1.0)**2

        # quantize error
        grad_input = quantization.quant_err(grad_input)

        return grad_input

superspike = SuperSpike().apply

class QLinearFunctional(torch.autograd.Function):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    @staticmethod
    def forward(context, input, weight, bias=None):
        input[input > 0]  = 1 #correct for dropout scale
        input[input <= 0] = 0
        context.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, bias = context.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # quantize error - this function should receive a quant error***
        # 2 bits (-1, 1) * 1 bit (0, 1/spikes)
        if context.needs_input_grad[0]:
            grad_input = quantization.quant_err(grad_output.mm(weight))

        # those weights should not be updated
        if context.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = torch.zeros_like(bias)

        return grad_input, grad_weight, grad_bias

class QLinearLayerSign(nn.Module):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    def __init__(self, input_features, output_features):
        super(QLinearLayerSign, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # weight and bias for forward pass
        self.weights = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weights.data.uniform_(-1, 1)

        self.weights.data[self.weights.data > 0] = 1
        self.weights.data[self.weights.data < 0] = -1
        

    def forward(self, input):
        return QLinearFunctional.apply(input, self.weights, None)


class QSLinearFunctional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, scale):
        w_quant = quantization.quant_w(weights, scale)
        bias_quant = quantization.quant_w(bias, scale)

        
        output = torch.einsum("ab,bc->ac", (input, w_quant)) + bias_quant
        
        ctx.save_for_backward(input, w_quant, bias_quant)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w_quant, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        quant_error = quantization.quant_err(grad_output) 

        # compute quantized error
        if ctx.needs_input_grad[0]:
            grad_input = torch.einsum("ab,cb->ac", (quant_error, w_quant))
        # computed quantized gradient
        if ctx.needs_input_grad[1]:
            grad_weight = quantization.quant_grad(torch.einsum("ab,ac->bc", (quant_error, input))).float()
        # computed quantized bias
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = quantization.quant_grad(quant_error.sum(0).squeeze(0)).float()


        return grad_input, grad_weight.T, grad_bias, None

class LIFDenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, tau_syn, tau_mem, tau_ref, delta_t, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float):
        super(LIFDenseLayer, self).__init__()    
        self.device = device  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.thr = thr
        self.fan_in = in_channels
        self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_wb)]))
        self.L = np.max([np.sqrt( 6/self.fan_in), self.L_min])
        self.scale = 2 ** round(math.log(self.L_min / self.L, 2.0))
        self.scale = self.scale if self.scale > 1 else 1.0

        if tau_syn.shape[0] == 2:
            self.beta = torch.exp( -delta_t / torch.Tensor(self.in_channels).uniform_(tau_syn[0], tau_syn[0]).to(device))
        else:
            self.beta = torch.Tensor([torch.exp( - delta_t / tau_syn)]).to(device)
        if tau_mem.shape[0] == 2:
            self.alpha = torch.exp( -delta_t / torch.Tensor(self.in_channels).uniform_(tau_mem[0], tau_mem[0]).to(device))
        else:
            self.alpha = torch.Tensor([torch.exp( - delta_t / tau_mem)]).to(device)

        if tau_ref.shape[0] == 2:
            self.gamma = torch.exp( -delta_t / torch.Tensor(self.out_channels).uniform_(tau_ref[0], tau_ref[0]).to(device))
        else:
            self.gamma = torch.Tensor([torch.exp( - delta_t / tau_ref)]).to(device)


        self.weights = nn.Parameter(torch.empty((self.in_channels, self.out_channels),  device=device, dtype=dtype, requires_grad=True))
        torch.nn.init.uniform_(self.weights, a = -self.L, b = self.L)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels,  device=device, dtype=dtype, requires_grad=True))
            torch.nn.init.uniform_(self.bias, a = -self.L, b = self.L)
        else:
            self.register_parameter('bias', None)

        with torch.no_grad():
            self.weights.data = quantization.clip(quantization.quant_generic(self.weights.data, quantization.global_gb)[0], quantization.global_wb)
            if self.bias is not None:
                self.bias.data = quantization.clip(quantization.quant_generic(self.bias.data, quantization.global_gb)[0], quantization.global_wb)

    def state_init(self, batch_size):
        self.P = torch.zeros(batch_size, self.in_channels).detach().to(self.device)
        self.Q = torch.zeros(batch_size, self.in_channels).detach().to(self.device)
        self.R = torch.zeros(batch_size, self.out_channels).detach().to(self.device)
        self.S = torch.zeros(batch_size, self.out_channels).detach().to(self.device)
        self.U = torch.zeros(batch_size, self.out_channels).detach().to(self.device)

    
    def forward(self, input_t):
        with torch.no_grad():
            self.weights.data = quantization.clip(self.weights.data, quantization.global_gb)
            if self.bias is not None:
                self.bias.data = quantization.clip(self.bias.data, quantization.global_gb)

        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t

        # quantize P, 
        self.P, _ = quantization.quant_generic(self.P, quantization.global_pb)
        self.Q, _ = quantization.quant_generic(self.Q, quantization.global_qb)
        

        self.U = QSLinearFunctional.apply(self.P, self.weights, self.bias, self.scale) + self.R
        self.S = (self.U > self.thr).float()

        # quantize U
        self.U, _ = quantization.quant_generic(self.U, quantization.global_ub)

        return self.S


class QSConv2dFunctional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, scale, padding = 0, pooling = None):
        w_quant = quantization.quant_w(weights, scale)
        bias_quant = quantization.quant_w(bias, scale)
        ctx.padding = padding
        ctx.pooling = pooling 
        ctx.size_pool = None
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
        quant_error = quantization.quant_err(grad_output)

        # compute quantized error
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, w_quant, quant_error, padding = ctx.padding)
        # computed quantized gradient
        if ctx.needs_input_grad[1]:
            grad_weight = quantization.quant_grad(torch.nn.grad.conv2d_weight(input, w_quant.shape, quant_error, padding = ctx.padding)).float()
        # computed quantized bias
        if bias_quant is not None and ctx.needs_input_grad[2]:
            grad_bias = quantization.quant_grad(torch.einsum("abcd->b",(quant_error))).float()

        return grad_input, grad_weight, grad_bias, None, None, None


class LIFConv2dLayer(nn.Module):
    def __init__(self, inp_shape, kernel_size, out_channels, tau_syn, tau_mem, tau_ref, delta_t, pooling = 1, padding = 0, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float):
        super(LIFConv2dLayer, self).__init__()   
        self.device = device
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.out_channels = out_channels  
        self.fan_in = kernel_size * kernel_size * inp_shape[0]
        self.L_min = quantization.global_beta/quantization.step_d(torch.tensor([float(quantization.global_wb)]))
        self.L = np.max([np.sqrt( 6/self.fan_in), self.L_min])
        self.scale = 2 ** round(math.log(self.L_min / self.L, 2.0))
        self.scale = self.scale if self.scale > 1 else 1.0

        self.padding = padding
        self.pooling = pooling
                
        self.weights = nn.Parameter(torch.empty((self.out_channels, inp_shape[0],  self.kernel_size, self.kernel_size),  device=device, dtype=dtype, requires_grad=True))
        torch.nn.init.uniform_(self.weights, a = -self.L, b = self.L)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype, requires_grad=True))
            torch.nn.init.uniform_(self.bias, a = -self.L, b = self.L)
        else:
            self.register_parameter('bias', None)

        self.out_shape = QSConv2dFunctional.apply(torch.zeros((1,)+self.inp_shape).to(device), self.weights, self.bias, self.scale, self.padding, self.pooling).shape[1:]
        self.thr = thr

        if tau_syn.shape[0] == 2:
            self.beta = torch.exp( -delta_t / torch.Tensor(self.inp_shape).uniform_(tau_syn[0], tau_syn[0]).to(device))
        else:
            self.beta = torch.Tensor([torch.exp( - delta_t / tau_syn)]).to(device)
        if tau_mem.shape[0] == 2:
            self.alpha = torch.exp( -delta_t / torch.Tensor(self.inp_shape).uniform_(tau_mem[0], tau_mem[0]).to(device))
        else:
            self.alpha = torch.Tensor([torch.exp( - delta_t / tau_mem)]).to(device)

        if tau_ref.shape[0] == 2:
            self.gamma = torch.exp( -delta_t / torch.Tensor(self.out_shape).uniform_(tau_ref[0], tau_ref[0]).to(device))
        else:
            self.gamma = torch.Tensor([torch.exp( - delta_t / tau_ref)]).to(device)

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

    
    def forward(self, input_t):
        with torch.no_grad():
            self.weights.data = quantization.clip(self.weights.data, quantization.global_gb)
            if self.bias is not None:
                self.bias.data = quantization.clip(self.bias.data, quantization.global_gb)

        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t

        # quantize P, Q
        self.P, _ = quantization.quant_generic(self.P, quantization.global_pb)
        self.Q, _ = quantization.quant_generic(self.Q, quantization.global_qb)

        self.U = QSConv2dFunctional.apply(self.P, self.weights, self.bias, self.scale, self.padding, self.pooling) + self.R
        self.S = (self.U > self.thr).float()

        # quantize U
        self.U, _ = quantization.quant_generic(self.U, quantization.global_ub)

        return self.S
