import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn

import line_profiler


#@profile
def compute_dropconnect_inputs(inputs, weights, p_drop, device):
    prob = 1-p_drop
    mu = prob * torch.einsum("abc,cd->abd", (inputs, weights))
    w_sqr = (weights*weights)
    inp_sqr = (inputs*inputs)
    sigma = prob*(1-prob) * torch.einsum("abc,cd->abd",  (inp_sqr, w_sqr))
    return torch.randn((sigma.shape[0], sigma.shape[1], sigma.shape[2]), device=device)*sigma+mu

#@profile
def input_compute(inputs, weights, layer, p_drop, device,  infer):
    if 'convolutional' in layer:
        # covolutional layer dont have dropconnect .... (?)
        nb_steps = inputs.shape[1]
        fin_inp = []

        for i in range(nb_steps):
            unfold_function = nn.Unfold(int(np.sqrt(weights.shape[0])),dilation=1,padding=0,stride=1)
            temp_inputs = inputs[:,i,:].reshape([inputs.shape[0],1,int(np.sqrt(inputs.shape[2])),int(np.sqrt(inputs.shape[2]))])
            fin_inp.append(unfold_function(temp_inputs))
        inputs = torch.stack(fin_inp, dim=1)
        return torch.einsum("abcd,ce->abd", (inputs,weights))


    if infer:
        return compute_dropconnect_inputs(inputs, weights, p_drop, device)
    else:
        mask = torch.bernoulli(1-p_drop * torch.ones_like(weights))
        return torch.einsum("abc,cd->abd", (inputs, weights*mask))

    return -1

#@profile
def LIF_neuron(inputs, weights, args, layer, layer_type, infer):
    tau_syn = args['tau_syn'][layer]
    tau_mem = args['tau_mem'][layer] 
    device  = args['device']
    spike_fn = args['spike_fn']
    time_step  = args['time_step']
    fire_thresh = args['fire_thresh'][layer]
    dtype = torch.float
    batch_size = inputs.shape[0]
    nb_steps = inputs.shape[1]


    h1 = input_compute(inputs, weights, layer_type, args['p_drop'], device, infer)
    nb_hidden = h1.shape[2]
    syn = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)

    alpha   = torch.exp(-time_step/tau_syn).repeat(syn.shape[0],1)
    beta    = torch.exp(-time_step/tau_mem).repeat(mem.shape[0],1)

    mem_rec = [mem]
    spk_rec = [mem]

    for t in range(nb_steps-1):
        mthr = mem-fire_thresh
        out = spike_fn(mthr)
        c = (mthr > 0)
        rst = torch.zeros_like(mem)
        rst[c] = torch.ones_like(mem)[c]

        syn, mem = alpha*syn + h1[:,t], beta*mem + syn -rst
        mem[c], syn[c] = 0, 0

        mem_rec.append(mem)
        spk_rec.append(out)

    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)

    import pdb; pdb.set_trace()
    return mem_rec, spk_rec

#@profile
def read_out_layer(inputs, weights, args, infer):
    if "no_spike" in args['read_out']:
        args_new = args.copy()
        args_new['fire_thresh'] = [torch.ones(weights.shape[1], device = args['device'], dtype = args['dtype'])*np.inf]
        mem_rec, spk_temp = args['neuron_type'](inputs=inputs, weights=weights, args = args_new, layer=-1, layer_type = "read_out", infer=infer)
        if args['read_out'] == "no_spike_max":
            m,_ = torch.max(mem_rec,1)
        if args['read_out'] == "no_spike_integrate":
            m = torch.sum(mem_rec,1)
    else:
        mem_rec, spk_temp = args['neuron_type'](inputs=inputs, weights=weights, args = args, layer=-1, layer_type = "read_out", infer=infer)
        if args['read_out'] == "spike_count":
            m = torch.sum(spk_temp,1)
        if args['read_out'] == "first_spike":
            #_, m = torch.max(spk_temp,1)
            #m[m==0] = spk_temp.shape[1]+10
            #m = 1/m.type( dtype = args['dtype'])

            ind_mask = torch.ones_like(spk_temp)
            ind_mask = torch.cumsum(ind_mask, dim=1)
            spk_temp = ind_mask * spk_temp
            spk_temp[spk_temp==0] = spk_temp.shape[1]+10
            m, _ = torch.min(spk_temp,1)
            m = 1/m
        if args['read_out'] == "avg_interval":
            import pdb; pdb.set_trace()
            m = 0 #not yet implemented 

    return m


def adex_LIF_neuron(inputs, weights, args, layer, layer_type, infer):
    tau_syn = args['tau_syn'][layer]
    tau_mem = args['tau_mem'][layer]
    tau_cur = args['tau_cur'][layer]
    sharpness = args['sharpness'][layer]
    a_cur = args['a_cur'][layer]
    b_cur = args['b_cur'][layer]
    theta = args['theta'][layer]
    device  = args['device']
    spike_fn = args['spike_fn']
    time_step  = args['time_step']
    dtype = torch.float
    batch_size = inputs.shape[0]
    nb_steps = inputs.shape[1]

    h1 = input_compute(inputs, weights, layer_type, args['p_drop'], device, infer)
    nb_hidden = h1.shape[2]
    syn = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    cur = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)

    # make sure that all dimensions fit, and its really a elementwise mult
    alpha     = torch.exp(-time_step/tau_syn).repeat(syn.shape[0],1)
    beta      = torch.exp(-time_step/tau_mem).repeat(mem.shape[0],1)
    gamma     = torch.exp(-time_step/tau_cur).repeat(cur.shape[0],1)
    a_cur     = a_cur.repeat(mem.shape[0],1)
    b_cur     = b_cur.repeat(mem.shape[0],1)
    sharpness = sharpness.repeat(mem.shape[0],1)

    mem_rec = [mem]
    spk_rec = [mem]
    cur_rec = [mem]    

    for t in range(nb_steps-1):
        # check for spike and reset to zero
        mthr = mem-1.0
        out = spike_fn(mthr)
        rst = torch.zeros_like(mem)
        c   = (mthr > 0)
        rst[c] = torch.ones_like(mem)[c]
        
        # neural dynamics, discretized diff. eqs.
        syn, mem, cur = alpha * syn + h1[:,t], beta * mem + syn + sharpness * torch.exp(torch.clamp((mem - theta)/sharpness, max = 80)) - cur -rst, gamma * cur + a_cur * mem + b_cur * rst
        mem[c], syn[c], cur[c] = 0, 0, 0 

        # recording over time
        mem_rec.append(mem)
        spk_rec.append(out)
        cur_rec.append(cur)

    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)
    cur_rec = torch.stack(cur_rec,dim=1)

    return mem_rec, spk_rec



def ferro_neuron(inputs, weights, args, layer, layer_type, infer):
    v_rest_e = args['v_rest_e'][layer]
    v_reset_e = args['v_reset_e'][layer]
    v_thresh_e = args['v_thresh_e'][layer]
    refrac_e = args['refrac_e'][layer]
    tau_v = args['tau_v'][layer]
    del_theta = args['del_theta'][layer]
    ge_max = args['ge_max'][layer]
    gi_max = args['gi_max'][layer]
    tau_ge = args['tau_ge'][layer]
    tau_gi = args['tau_gi'][layer]
    device  = args['device']
    spike_fn = args['spike_fn']
    time_step  = args['time_step']
    dx_dt_param = args['dx_dt_param'][layer]
    dtype = torch.float
    batch_size = inputs.shape[0]
    nb_steps = inputs.shape[1]


    h1 = input_compute(inputs, weights, layer_type, args['p_drop'], device, infer)
    nb_hidden = h1.shape[2]
    g_e = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    theta = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    I_syn_E = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    v = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype) * v_rest_e
    dx_dt = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    alpha = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype)
    v_rest_e = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype) * v_rest_e
    del_theta = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype) * del_theta

    v_rec       = [v]
    spk_rec     = [g_e]
    
    for t in range(nb_steps-1):
        dge_dt = -g_e/tau_ge
        g_e, I_syn_E, dx_dt = g_e + time_step*dge_dt + h1[:,t], (- g_e*v)*1e-9, -g_e/dx_dt_param
        alpha = alpha + time_step * dx_dt
        dv_dt = (v_rest_e*alpha - v)/(.1*tau_v) + (I_syn_E/1e-9)/(1*tau_v)
        v = v + time_step*dv_dt

        out = spike_fn(v - (v_thresh_e+theta))
        c = (out==1.0)
        v[c] = v_rest_e[c]
        theta[c] += del_theta[c]
        alpha[c] = 1

        v_rec.append(v)
        spk_rec.append(out)
        

    v_rec   = torch.stack(v_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    #import pdb; pdb.set_trace()
    return v_rec, spk_rec


def ferroLIF_neuron(inputs, weights, args, layer, layer_type, infer):
    v_rest_e = args['v_rest_e'][layer]
    v_reset_e = args['v_reset_e'][layer]
    v_thresh_e = args['v_thresh_e'][layer]
    refrac_e = args['refrac_e'][layer]
    tau_v = args['tau_v'][layer]
    del_theta = args['del_theta'][layer]
    ge_max = args['ge_max'][layer]
    gi_max = args['gi_max'][layer]
    tau_ge = args['tau_ge'][layer]
    tau_gi = args['tau_gi'][layer]
    device  = args['device']
    spike_fn = args['spike_fn']
    time_step  = args['time_step']
    dx_dt_param = args['dx_dt_param'][layer]
    dtype = torch.float
    batch_size = inputs.shape[0]
    nb_steps = inputs.shape[1]


    h1 = input_compute(inputs, weights, layer_type, args['p_drop'], device, infer)
    nb_hidden = h1.shape[2]
    g_e = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    theta = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    I_syn_E = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    v = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype) * v_rest_e
    dx_dt = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    alpha = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype)
    v_rest_e = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype) * v_rest_e
    del_theta = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype) * del_theta

    v_rec       = [v]
    spk_rec     = [g_e]
    
    for t in range(nb_steps-1):
        dge_dt = -g_e/tau_ge
        g_e, I_syn_E = g_e + time_step*dge_dt + h1[:,t], (- g_e*v)*1e-9
        dv_dt = (v_rest_e - v)/tau_v + (I_syn_E/1e-9)/(1*tau_v)
        v = v + time_step*dv_dt

        out = spike_fn(v - (v_thresh_e+theta))
        c = (out==1.0)
        v[c] = v_rest_e[c]
        theta[c] += del_theta[c]
        alpha[c] = 1

        v_rec.append(v)
        spk_rec.append(out)
        

    v_rec   = torch.stack(v_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    #import pdb; pdb.set_trace()
    return v_rec, spk_rec