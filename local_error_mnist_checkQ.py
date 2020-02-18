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

        # check whether quantized weights are really quantized
        if np.logical_not(np.isin(grad_input.cpu(), quantization.valid_e_vals.cpu())).sum() != 0:
            print('Error not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_e_vals += torch.cat([grad_input.flatten().cpu(), quantization.valid_e_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_e_vals, dtype= int)

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

        # check whether quantized weights are really quantized
        if np.logical_not(np.isin(grad_input.cpu(), quantization.valid_e_vals.cpu())).sum() != 0:
            print('Error not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_e_vals += torch.cat([grad_input.flatten().cpu(), quantization.valid_e_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_e_vals, dtype= int)

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

            # check whether quantized weights are really quantized
            if np.logical_not(np.isin(grad_input.cpu(), quantization.valid_e_vals.cpu())).sum() != 0:
                print('Error not properly quantized')
                import pdb; pdb.set_trace()
            else:
                quantization.count_e_vals += torch.cat([grad_input.flatten().cpu(), quantization.valid_g_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_e_vals, dtype= int)

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

        # check whether quantized weights are really quantized
        if np.logical_not(np.isin(w_quant.cpu(), quantization.valid_w_vals.cpu())).sum() != 0:
            print('Weights not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_w_vals += torch.cat([w_quant.flatten().cpu(), quantization.valid_w_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_w_vals, dtype= int)
        
        output = torch.einsum("ab,bc->ac", (input, w_quant)) + bias_quant
        
        ctx.save_for_backward(input, w_quant, bias_quant)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w_quant, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        quant_error = quantization.quant_err(grad_output) 

        # check whether quantized weights are really quantized
        if np.logical_not(np.isin(quant_error.cpu(), quantization.valid_e_vals.cpu())).sum() != 0:
            print('Error not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_e_vals += torch.cat([quant_error.flatten().cpu(), quantization.valid_e_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_e_vals, dtype= int)

        # compute quantized error
        if ctx.needs_input_grad[0]:
            grad_input = torch.einsum("ab,cb->ac", (quant_error, w_quant))
        # computed quantized gradient
        if ctx.needs_input_grad[1]:
            grad_weight = quantization.quant_grad(torch.einsum("ab,ac->bc", (quant_error, input))).float()
            #grad_weight = torch.einsum("ab,ac->bc", (quant_error, input))
        # computed quantized bias
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = quantization.quant_grad(quant_error.sum(0).squeeze(0)).float()
            #grad_bias = quant_error.sum(0).squeeze(0)

        # check whether quantized weights are really quantized
        if (grad_weight % (1/quantization.step_d(quantization.global_gb))).sum() != 0:
            print('Gradient not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_g_vals += torch.cat([quantization.clip(grad_weight.flatten().cpu(), quantization.global_gb), quantization.valid_g_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_g_vals, dtype= int)

        return grad_input, grad_weight.T, grad_bias, None

class LIFDenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, tau_syn, tau_mem, tau_ref, delta_t, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float):
        super(LIFDenseLayer, self).__init__()      
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
        self.P = torch.zeros(batch_size, self.in_channels).detach().to(device)
        self.Q = torch.zeros(batch_size, self.in_channels).detach().to(device)
        self.R = torch.zeros(batch_size, self.out_channels).detach().to(device)
        self.S = torch.zeros(batch_size, self.out_channels).detach().to(device)
        self.U = torch.zeros(batch_size, self.out_channels).detach().to(device)

    
    def forward(self, input_t):
        with torch.no_grad():
            #self.weights.data = quantization.clip(quantization.quant_generic(self.weights.data, quantization.global_gb)[0], quantization.global_wb)
            #self.bias.data = quantization.clip(quantization.quant_generic(self.bias.data, quantization.global_gb)[0], quantization.global_wb)

            self.weights.data = quantization.clip(self.weights.data, quantization.global_gb)
            if self.bias is not None:
                self.bias.data = quantization.clip(self.bias.data, quantization.global_gb)

            # check whether quantized weights are really quantized
            if np.logical_not(np.isin(self.weights.data.cpu(), quantization.valid_g_vals.cpu())).sum() != 0:
                print('Stored Weights not properly quantized')
                import pdb; pdb.set_trace()
            else:
                quantization.count_g_vals += torch.cat([self.weights.data.flatten().cpu(), quantization.valid_g_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_g_vals, dtype= int)


        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t

        # quantize P, 
        self.P, _ = quantization.quant_generic(self.P, quantization.global_pb)
        self.Q, _ = quantization.quant_generic(self.Q, quantization.global_qb)
        

        self.U = QSLinearFunctional.apply(self.P, self.weights, self.bias, self.scale) + self.R
        self.S = (self.U > self.thr).float()

        # quantize U
        self.U, _ = quantization.quant_generic(self.U, quantization.global_ub)

        # check whether quantized weights are really quantized
        if np.logical_not(np.isin(self.P.cpu(), quantization.valid_p_vals.cpu())).sum() != 0:
            print('P not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_p_vals += torch.cat([self.P.flatten().cpu(), quantization.valid_p_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_p_vals, dtype= int)

        # check whether quantized weights are really quantized
        if np.logical_not(np.isin(self.Q.cpu(), quantization.valid_q_vals.cpu())).sum() != 0:
            print('Q not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_q_vals += torch.cat([self.Q.flatten().cpu(), quantization.valid_q_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_q_vals, dtype= int)

        # check whether quantized weights are really quantized
        dummy = self.U.clone().detach()
        if np.logical_not(np.isin(dummy.cpu(), quantization.valid_u_vals.cpu())).sum() != 0:
            print('U not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_u_vals += torch.cat([self.U.flatten().cpu(), quantization.valid_u_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_u_vals, dtype= int)


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

        # check whether quantized weights are really quantized
        if np.logical_not(np.isin(w_quant.cpu(), quantization.valid_w_vals.cpu())).sum() != 0:
            print('Weights not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_w_vals += torch.cat([w_quant.flatten().cpu(), quantization.valid_w_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_w_vals, dtype = int)

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

        # check whether quantized weights are really quantized
        if np.logical_not(np.isin(quant_error.cpu(), quantization.valid_e_vals.cpu())).sum() != 0:
            print('Error not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_e_vals += torch.cat([quant_error.flatten().cpu(), quantization.valid_e_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_e_vals, dtype= int)

        # compute quantized error
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, w_quant, quant_error, padding = ctx.padding)
        # computed quantized gradient
        if ctx.needs_input_grad[1]:
            grad_weight = quantization.quant_grad(torch.nn.grad.conv2d_weight(input, w_quant.shape, quant_error, padding = ctx.padding)).float()
            #grad_weight = torch.nn.grad.conv2d_weight(input, w_quant.shape, quant_error, padding = ctx.padding)
        # computed quantized bias
        if bias_quant is not None and ctx.needs_input_grad[2]:
            grad_bias = quantization.quant_grad(torch.einsum("abcd->b",(quant_error))).float()
            #grad_bias = torch.einsum("abcd->b",(quant_error))

        # check whether quantized weights are really quantized
        if (grad_weight % (1/quantization.step_d(quantization.global_gb))).sum() != 0:
            print('Gradient not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_g_vals += torch.cat([quantization.clip(grad_weight.flatten().cpu(), quantization.global_gb), quantization.valid_g_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_g_vals, dtype= int)

        return grad_input, grad_weight, grad_bias, None, None, None


class LIFConv2dLayer(nn.Module):
    def __init__(self, inp_shape, kernel_size, out_channels, tau_syn, tau_mem, tau_ref, delta_t, pooling = 1, padding = 0, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float):
        super(LIFConv2dLayer, self).__init__()   
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
        self.P = torch.zeros((batch_size,) + self.inp_shape).detach().to(device)
        self.Q = torch.zeros((batch_size,) + self.inp_shape).detach().to(device)
        self.R = torch.zeros((batch_size,) + self.out_shape).detach().to(device)
        self.S = torch.zeros((batch_size,) + self.out_shape).detach().to(device)
        self.U = torch.zeros((batch_size,) + self.out_shape).detach().to(device)

    
    def forward(self, input_t):
        with torch.no_grad():
            self.weights.data = quantization.clip(self.weights.data, quantization.global_gb)
            if self.bias is not None:
                self.bias.data = quantization.clip(self.bias.data, quantization.global_gb)
            
            # check whether quantized weights are really quantized
            if np.logical_not(np.isin(self.weights.data.cpu(), quantization.valid_g_vals.cpu())).sum() != 0:
                print('Stored Weights not properly quantized')
                import pdb; pdb.set_trace()
            else:
                quantization.count_g_vals += torch.cat([self.weights.data.flatten().cpu(), quantization.valid_g_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_g_vals, dtype= int)

        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t

        # quantize P, Q
        self.P, _ = quantization.quant_generic(self.P, quantization.global_pb)
        self.Q, _ = quantization.quant_generic(self.Q, quantization.global_qb)

        self.U = QSConv2dFunctional.apply(self.P, self.weights, self.bias, self.scale, self.padding, self.pooling) + self.R
        self.S = (self.U > self.thr).float()

        # quantize U
        self.U, _ = quantization.quant_generic(self.U, quantization.global_ub)

        # check whether quantized weights are really quantized
        if np.logical_not(np.isin(self.P.cpu(), quantization.valid_p_vals.cpu())).sum() != 0:
            print('P not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_p_vals += torch.cat([self.P.flatten().cpu(), quantization.valid_p_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_p_vals, dtype= int)

        # check whether quantized weights are really quantized
        if np.logical_not(np.isin(self.Q.cpu(), quantization.valid_q_vals.cpu())).sum() != 0:
            print('Q not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_q_vals += torch.cat([self.Q.flatten().cpu(), quantization.valid_q_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_q_vals, dtype= int)

        # check whether quantized weights are really quantized
        dummy = self.U.clone().detach()
        if np.logical_not(np.isin(dummy.cpu(), quantization.valid_u_vals.cpu())).sum() != 0:
            print('U not properly quantized')
            import pdb; pdb.set_trace()
        else:
            quantization.count_u_vals += torch.cat([self.U.flatten().cpu(), quantization.valid_u_vals]).unique(return_counts = True)[1] - torch.ones_like(quantization.count_u_vals, dtype= int)

        return self.S




# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float


train_dataset = torchvision.datasets.MNIST('../data', train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.MNIST('../data', train=False, transform=None, target_transform=None, download=True)

# Standardize data
x_train = train_dataset.data.type(dtype)/255
x_train = x_train.reshape((x_train.shape[0],) + (1,) + x_train.shape[1:])
x_test = test_dataset.data.type(dtype)/255
x_test = x_test.reshape((x_test.shape[0],) + (1,) + x_test.shape[1:])

y_train = train_dataset.targets
y_test  = test_dataset.targets

shuffle_idx = torch.randperm(len(y_train))
x_train = x_train[shuffle_idx]
y_train = y_train[shuffle_idx]
shuffle_idx = torch.randperm(len(y_test))
x_test = x_test[shuffle_idx]
y_test = y_test[shuffle_idx]

# fixed subsampling
# train: 300 samples per class -> 3000
# test: 103 samples per class -> 1030 (a wee more than 1024)
train_samples = 1000
test_samples = 500
num_classes = 10
index_list_train = []
index_list_test = []
for i in range(10):
    index_list_train.append((y_train == i).nonzero()[:int(train_samples/num_classes)])
    index_list_test.append((y_test == i).nonzero()[:int(test_samples/num_classes)])
index_list_train = torch.cat(index_list_train).reshape([train_samples])
index_list_test = torch.cat(index_list_test).reshape([test_samples])

x_train = x_train[index_list_train, :]
x_test = x_test[index_list_test, :]
y_train = y_train[index_list_train]
y_test = y_test[index_list_test]


#quantization.global_beta = 1.5
quantization.global_wb = 8
quantization.global_ub = 8
quantization.global_qb = 8
quantization.global_pb = 8
quantization.global_gb = 8
quantization.global_eb = 8
quantization.global_rb = 16
quantization.global_lr = 1
quantization.global_beta = 1.5 #quantization.step_d(quantization.global_wb)-.5
# effect of global beta 


# visualize quant
quantization.valid_w_vals = quantization.quant_w(torch.cat([torch.arange(-1, 1, 2/((2**quantization.global_wb)-1)), torch.tensor([0.])]), 1).unique()
quantization.count_w_vals = torch.zeros_like(quantization.valid_w_vals, dtype = int)

quantization.valid_p_vals = quantization.quant_generic(torch.cat([torch.arange(-1, 1, 2/((2**quantization.global_pb)-1)), torch.tensor([0.])]), quantization.global_pb)[0].unique()
quantization.count_p_vals = torch.zeros_like(quantization.valid_p_vals, dtype = int)

quantization.valid_q_vals = quantization.quant_generic(torch.cat([torch.arange(-1, 1, 2/((2**quantization.global_qb)-1)), torch.tensor([0.])]), quantization.global_qb)[0].unique()
quantization.count_q_vals = torch.zeros_like(quantization.valid_q_vals, dtype = int)

quantization.valid_u_vals = quantization.quant_generic(torch.cat([torch.arange(-1, 1, 2/((2**quantization.global_ub)-1)), torch.tensor([0.])]), quantization.global_ub)[0].unique()
quantization.count_u_vals = torch.zeros_like(quantization.valid_u_vals, dtype = int)

quantization.valid_e_vals = quantization.quant_generic(torch.cat([torch.arange(-1, 1, 2/((2**quantization.global_eb)-1)), torch.tensor([0.])]), quantization.global_eb)[0].unique()
quantization.count_e_vals = torch.zeros_like(quantization.valid_e_vals, dtype = int)

quantization.valid_g_vals = quantization.quant_generic(torch.cat([torch.arange(-1, 1, 2/((2**quantization.global_gb)-1)), torch.tensor([0.])]), quantization.global_gb)[0].unique()
quantization.count_g_vals = torch.zeros_like(quantization.valid_g_vals, dtype = int)


ms = 1e-3
delta_t = 1*ms

T = 500*ms
T_test = 1000*ms
burnin = 50*ms
batch_size = 128
output_neurons = 10

tau_mem = torch.Tensor([5*ms, 35*ms]).to(device)
tau_syn = torch.Tensor([5*ms, 10*ms]).to(device)
tau_ref = torch.Tensor([0*ms]).to(device)
thr = torch.Tensor([.4]).to(device)

lambda1 = .2 
lambda2 = .1

dropout_learning = nn.Dropout(p=.5)

layer1 = LIFConv2dLayer(inp_shape = x_train.shape[1:], kernel_size = 7, out_channels = 16, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout1 = QLinearLayerSign(np.prod(layer1.out_shape), output_neurons).to(device)

layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape, kernel_size = 7, out_channels = 24, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 1, padding = 2, thr = thr, device = device).to(device)
random_readout2 = QLinearLayerSign(np.prod(layer2.out_shape), output_neurons).to(device)

layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape, kernel_size = 7, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout3 = QLinearLayerSign(np.prod(layer3.out_shape), output_neurons).to(device)

layer4 = LIFDenseLayer(in_channels = np.prod(layer3.out_shape), out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)

log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
nll_loss = torch.nn.NLLLoss()

# shall I train this every time upfront?
# global_lr = 3.3246e-4
# opt1 = torch.optim.Adam(layer1.parameters(), lr=global_lr, betas=[0., .95])
# opt2 = torch.optim.Adam(layer2.parameters(), lr=global_lr, betas=[0., .95])
# opt3 = torch.optim.Adam(layer3.parameters(), lr=global_lr, betas=[0., .95])
# opt4 = torch.optim.Adam(layer4.parameters(), lr=global_lr, betas=[0., .95])

opt1 = torch.optim.SGD(layer1.parameters(), lr=1)
opt2 = torch.optim.SGD(layer2.parameters(), lr=1)
opt3 = torch.optim.SGD(layer3.parameters(), lr=1)
opt4 = torch.optim.SGD(layer4.parameters(), lr=1)
scheduler1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=20, gamma=0.5)
scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=20, gamma=0.5)
scheduler3 = torch.optim.lr_scheduler.StepLR(opt3, step_size=20, gamma=0.5)
scheduler4 = torch.optim.lr_scheduler.StepLR(opt4, step_size=20, gamma=0.5)

print("WPQUEG Quantization: {0}".format(quantization.global_wb))

for e in range(1):
    correct = 0
    total = 0
    tcorrect = 0
    ttotal = 0
    loss_hist = []
    start_time = time.time()

    for x_local, y_local in sparse_data_generator(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, samples = train_samples, max_hertz = 50, shuffle = True, device = device):
        class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)

        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])
        layer4.state_init(x_local.shape[0])

        # burnin
        for t in range(int(burnin/ms)):
            out_spikes1 = layer1.forward(x_local[:,:,:,:,t])
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)

        # training
        for t in range(int(burnin/ms), int(T/ms)):
            out_spikes1 = layer1.forward(x_local[:,:,:,:,t])
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
            class_rec += out_spikes4
            print(loss_t4.item())

        correct += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        total += len(y_local)
    train_time = time.time()


    # compute test accuracy
    for x_local, y_local in sparse_data_generator(x_test, y_test, batch_size = batch_size, nb_steps = T_test/ms, samples = test_samples, max_hertz = 50, shuffle = True, device = device):
        class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)
        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])
        layer4.state_init(x_local.shape[0])

        for t in range(int(T_test/ms)):
            out_spikes1 = layer1.forward(x_local[:,:,:,:,t])
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)
            class_rec += out_spikes4
        tcorrect += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        ttotal += len(y_local)
    inf_time = time.time()

    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()

    print("Epoch {0} | Loss: {1:.4f} Train Acc: {2:.4f} Test Acc: {3:.4f} Train Time: {4:.4f}s Inference Time: {5:.4f}s".format(e+1, np.mean(loss_hist), correct.item()/total, tcorrect.item()/ttotal, train_time-start_time, inf_time - train_time)) 



# ta1500, te800, b128
# Epoch 1 | Loss: 1.6959 Train Acc: 0.8993 Test Acc: 0.9113 Train Time: 92.4179s Inference Time: 45.9957s
# Epoch 1 | Loss: 1.6540 Train Acc: 0.9747 Test Acc: 0.9287 Train Time: 93.9506s Inference Time: 45.2598s

# ta1000, te500, b128
# Epoch 1 | Loss: 1.6940 Train Acc: 0.9460 Test Acc: 0.8960 Train Time: 62.1373s Inference Time: 27.2974s

results = {'W': [quantization.valid_w_vals, quantization.count_w_vals], 'P': [quantization.valid_p_vals, quantization.count_p_vals], 'Q': [quantization.valid_q_vals, quantization.count_q_vals], 'U': [quantization.valid_u_vals, quantization.count_u_vals], 'E': [quantization.valid_e_vals, quantization.count_e_vals], 'G': [quantization.valid_g_vals, quantization.count_g_vals]}

with open('results/quant_check.pkl', 'wb') as f:
    pickle.dump(results, f)