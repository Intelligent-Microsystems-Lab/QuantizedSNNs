import os

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pickle

import torch
import torch.nn as nn
import torchvision

import spytorch_util
import quantization
from quantization import quant_act, init_layer_weights, SSE, to_cat, clip, quant_w, quant_err, quant_grad, quant_generic, step_d

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

def conv_exp_kernel(inputs, time_step, tau, device):
    dtype = torch.float
    nb_hidden = inputs.shape[1]
    nb_steps = inputs.shape[0]
    # hyper para -> what value do we need here?
    #alpha = float(torch.exp(torch.tensor([-time_step/tau], device= device, dtype=dtype)))

    u = torch.zeros((nb_hidden), device=device, dtype=dtype)
    rec_u = []
    
    for t in range(nb_steps):
        u = alpha*u + inputs[t,:]
        rec_u.append(u)

    rec_u = torch.stack(rec_u,dim=0)    
    return rec_u

def van_rossum(x, y, time_step, tau, device):
    tild_x = conv_exp_kernel(x, time_step, tau, device)
    tild_y = conv_exp_kernel(y, time_step, tau, device)
    return torch.sqrt(1/tau*torch.sum((tild_x - tild_y)**2))

class SuperSpike(torch.autograd.Function):
    scale = 100.0 # controls steepness of surrogate gradient
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SuperSpike.scale*torch.abs(input)+1.0)**2
        return grad


class einsum_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, scale, bias=None):
        if quantization.global_wb < stop_quant_level:
            w_quant = quant_w(weight, scale)
        else:
            w_quant = weight

        h1 = torch.einsum("bc,cd->bd", (input, w_quant))
        
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, w_quant, bias)

        return h1

    @staticmethod
    def backward(ctx, grad_output):
        input, w_quant, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if quantization.global_eb < stop_quant_level:
            quant_error = quant_err(grad_output)
        else:
            quant_error = grad_output

        if ctx.needs_input_grad[0]:
            # propagate quantized error
            grad_input = torch.einsum("bc,dc->bd", (quant_error, w_quant))

        if ctx.needs_input_grad[1]:
            if quantization.global_gb < stop_quant_level:
                grad_weight = quant_grad(torch.einsum("bc,bd->dc", (quant_error, input))).float()
            else:
                grad_weight = torch.einsum("bc,bd->dc", (quant_error, input))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class custom_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, b_level):
        if quantization.global_ab < stop_quant_level:
            output, clip_info = quant_act(input)
        else:
            output, clip_info = input, None
        ctx.save_for_backward(clip_info)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        clip_info = ctx.saved_tensors
        if quantization.global_eb < stop_quant_level:
            quant_error = quant_err(grad_output) * clip_info[0].float()
        else:
            quant_error = grad_output
        return quant_error, None


def run_snn(inputs):
    with torch.no_grad():
        spytorch_util.w1.data = clip(spytorch_util.w1.data, quantization.global_wb)
        spytorch_util.w2.data = clip(spytorch_util.w2.data, quantization.global_wb)


    h1 = einsum_linear.apply(inputs, spytorch_util.w1, scale1)

    syn = torch.zeros((nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((nb_hidden), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    for t in range(nb_steps):
        mthr = mem-.9
        mthr = custom_quant.apply(mthr, quantization.global_ab)
        out = spike_fn(mthr)

        rst = torch.zeros_like(mem)
        c   = (mthr > 0)
        rst[c] = torch.ones_like(mem)[c]

        new_syn = alpha*syn +h1[t,:]
        new_syn = custom_quant.apply(new_syn, quantization.global_ab)
        new_mem = beta*mem +syn -rst
        new_mem = custom_quant.apply(new_mem, quantization.global_ab)

        syn = new_syn
        mem = new_mem

        mem_rec.append(mem)
        spk_rec.append(out)

    mem_rec1 = torch.stack(mem_rec,dim=0)
    spk_rec1 = torch.stack(spk_rec,dim=0)


    #Readout layer
    h2 = einsum_linear.apply(spk_rec1, spytorch_util.w2, scale2)

    syn = torch.zeros((nb_outputs), device=device, dtype=dtype)
    mem = torch.zeros((nb_outputs), device=device, dtype=dtype)
    
    mem_rec = []
    spk_rec = []
    
    for t in range(nb_steps):
        mthr = mem-.9
        mthr = custom_quant.apply(mthr, quantization.global_ab)
        out = spike_fn(mthr)

        rst = torch.zeros_like(mem)
        c   = (mthr > 0)
        rst[c] = torch.ones_like(mem)[c]

        new_syn = alpha*syn +h2[t,:]
        new_syn = custom_quant.apply(new_syn, quantization.global_ab)
        new_mem = beta*mem +syn -rst
        new_mem = custom_quant.apply(new_mem, quantization.global_ab)

        mem = new_mem 
        syn = new_syn

        mem_rec.append(mem)
        spk_rec.append(out)

    mem_rec2 = torch.stack(mem_rec,dim=0)
    spk_rec2 = torch.stack(spk_rec,dim=0)


    other_recs = [mem_rec1, spk_rec1, mem_rec2]
    return spk_rec2, other_recs


def train(x_data, y_data, lr=1e-3, nb_epochs=10):
    params = [spytorch_util.w1,spytorch_util.w2]
    optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))

    loss_hist = []
    acc_hist = []
    for e in range(nb_epochs):
        output,recs = run_snn(x_data)
        loss_val = van_rossum(output, y_data, time_step, tau_syn, device)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())
        print("Epoch %i: loss=%.5f"%(e+1,loss_val.item()))
        
    return loss_hist, output
       
spike_fn  = SuperSpike.apply 

quantization.global_wb = 2
quantization.global_ab = 8
quantization.global_gb = 8
quantization.global_eb = 8
quantization.global_rb = 16
stop_quant_level = 33


time_step = 1e-3
nb_steps  = 250
tau_mem = 10e-3
tau_syn = 5e-3
tau_vr  = 5e-3
lr_change = 2e-4



#alpha   = float(np.exp(-time_step/tau_syn))
alpha   = .75
#beta    = float(np.exp(-time_step/tau_mem))
beta    = .875


nb_inputs  = 700
nb_hidden  = 400
nb_outputs = 250


quantization.global_beta = quantization.step_d(quantization.global_wb)-.5
with open("./data/input_700_250_25.pkl", 'rb') as f:
    x_train = pickle.load(f).t().to(device)

#x_train = torch.tensor(np.array(x_train.cpu(), dtype=bool)).double().to(device)


with open("./data/nd70.pkl", 'rb') as f:
    y_train = torch.tensor(pickle.load(f)).to(device)
y_train = y_train.type(dtype)
#y_train = torch.tensor(np.array(y_train, dtype=bool)).double().to(device)

bit_string = str(quantization.global_wb) + str(quantization.global_ab) + str(quantization.global_gb) + str(quantization.global_eb)

print(bit_string)

spytorch_util.w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
scale1 = init_layer_weights(spytorch_util.w1, 28*28).to(device)

spytorch_util.w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
scale2 = init_layer_weights(spytorch_util.w2, 28*28).to(device)


quantization.global_lr = .1
# lr = 2e-4
loss_hist, output = train(x_train, y_train, lr = 1, nb_epochs = 20) #/step_d(16)*10

bit_string = str(quantization.global_wb) + str(quantization.global_ab) + str(quantization.global_gb) + str(quantization.global_eb)

results = {'bit_string': bit_string ,'loss_hist': loss_hist, 'output': output.cpu()}

with open('results/snn_nd_precise_'+bit_string+'.pkl', 'wb') as f:
    pickle.dump(results, f)



