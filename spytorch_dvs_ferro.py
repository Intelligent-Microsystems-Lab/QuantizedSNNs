import os
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pickle

import torch
import torch.nn as nn
import torchvision

import quantization
import spytorch_util
from quantization import quant_act, init_layer_weights, SSE, to_cat, clip, quant_w, quant_err, quant_grad, quant_generic, step_d
from spytorch_util import current2firing_time, sparse_data_generator, plot_voltage_traces, SuperSpike, sparse_data_generator_DVS




ap = argparse.ArgumentParser()
ap.add_argument("-wb", "--wb", type = int, help = "weight bits")
ap.add_argument("-m", "--m", type = int, help="multiplier")
ap.add_argument("-ds", "--ds", type = int, help="dataset")
args = vars(ap.parse_args())


quantization.global_wb = args['wb']
inp_mult = args['m']
select_ds = args['ds']

if quantization.global_wb == None:
    quantization.global_wb = 4

if inp_mult == None:
    inp_mult = 150

if select_ds == None:
    select_ds = "MNIST"


#here test

# set global variables
#quantization.global_wb = 4
quantization.global_ab = 33
quantization.global_gb = 33
quantization.global_eb = 33
quantization.global_rb = 16
quantization.global_lr = 1

nb_inputs  = 128*128
nb_hidden  = 500
nb_outputs = 12

time_step = 1e-3 
nb_steps  = 100

batch_size = 256
dtype = torch.float

stop_quant_level = 33


# LIF with those, no quant .95
tau_mem = 10e-3
tau_syn = 5e-3
alpha1   = float(np.exp(-time_step/tau_syn))
beta    = float(np.exp(-time_step/tau_mem))



# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")


# Here we load the Dataset

test_dataset = pd.read_pickle('../DVS/test_complete.pkl')
y_test = torch.tensor(test_dataset['label'], device=device, dtype=dtype)
train_dataset = pd.read_pickle('../DVS/train_complete.pkl')
y_train = torch.tensor(train_dataset['label'], device=device, dtype=dtype)
with open('../DVS_prep/full_data_train.pkl', 'rb') as f:
   train_data = pickle.load(f)
with open('../DVS_prep/full_data_test.pkl', 'rb') as f:
    test_data = pickle.load(f)
x_test = pd.DataFrame({'batch':test_data[0],'ts':test_data[1],'unit':test_data[2]})
x_test = x_test.drop_duplicates()
x_train = pd.DataFrame({'batch':train_data[0],'ts':train_data[1],'unit':train_data[2]})
x_train = x_train.drop_duplicates()


class einsum_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, scale, bias=None):
        if quantization.global_wb < stop_quant_level:
            w_quant = quant_w(weight, scale)
        else:
            w_quant = weight

        h1 = torch.einsum("abc,cd->abd", (input, w_quant))
        
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
            grad_input = torch.einsum("abc,dc->abd", (quant_error, w_quant))

        if ctx.needs_input_grad[1]:
            if quantization.global_gb < stop_quant_level:
                grad_weight = quant_grad(torch.einsum("abc,abd->dc", (quant_error, input))).float()
            else:
                grad_weight = torch.einsum("abc,abd->dc", (quant_error, input))

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



print("init done")
    
# here we overwrite our naive spike function by the "SuperSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SuperSpike.apply


def run_snn(inputs):


    mV = 1e-3
    ms = 1e-3
    nS = 1e-9

    # Neuron parameterss
    v_exc = 0*mV
    v_inh = -100*mV
    v_rest_e = -65*mV
    v_reset_e_mult = -65*mV
    v_thresh_e_mult = -52*mV
    refrac_e = 5*ms
    tau_v = 100*ms

    theta = 0
    del_theta_mult = 0.1*mV


    t_leak = 1
    tau_leak = 100

    # Synapse parameters
    ge_max = 8
    gi_max = 5
    tau_ge = 1*ms
    tau_gi = 2*ms


    # doesnt change anything but I'll keep it out for now
    with torch.no_grad():
        spytorch_util.w1.data = clip(spytorch_util.w1.data, quantization.global_wb)
        spytorch_util.w2.data = clip(spytorch_util.w2.data, quantization.global_wb)

    h1 = einsum_linear.apply(inputs, spytorch_util.w1, scale1)

    g_e = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    v = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype) * v_rest_e
    alpha = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    theta = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    v_thresh_e = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype)*v_thresh_e_mult
    del_theta = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype)*del_theta_mult
    v_reset_e = torch.ones((batch_size,nb_hidden), device=device, dtype=dtype)*v_reset_e_mult


    mem_rec = [v]
    spk_rec = [g_e]
    
    for t in range(nb_steps-1):
        dge_dt = -g_e/tau_ge


        g_e, I_syn_E, dx_dt = g_e + time_step*dge_dt + h1[:,t]*inp_mult, (g_e*v_exc - g_e*v)*nS, -g_e/200e-3
        alpha = alpha + time_step*dx_dt
        # ferro
        dv_dt = (v_rest_e*alpha - v)/(.1*tau_v) + (I_syn_E/nS)/(1*tau_v)
        # ferro lif
        #dv_dt = (v_rest_e - v)/(1*tau_v) + (I_syn_E/nS)/(1*tau_v)
        v = v + time_step*dv_dt

        out = spike_fn(v - (v_thresh_e+theta))
        c = (out==1.0)
        theta[c] += del_theta[c]
        alpha[c] = 1
        v[c] = v_reset_e[c]

        mem_rec.append(v)
        spk_rec.append(out)

    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)


    #import pdb; pdb.set_trace()

    # syn = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    # mem = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)#

    # mem_rec = [mem]
    # spk_rec = [mem]

    # #Compute hidden layer activity
    # for t in range(nb_steps):
    #     mthr = mem-.9
    #     mthr = custom_quant.apply(mthr, quantization.global_ab)
    #     out = spike_fn(mthr)

    #     rst = torch.zeros_like(mem)
    #     c   = (mthr > 0)
    #     rst[c] = torch.ones_like(mem)[c]

    #     new_syn = alpha1*syn +h1[:,t]
    #     #new_syn = custom_quant.apply(new_syn, quantization.global_ab)
    #     new_mem = beta*mem +syn -rst
    #     #new_mem = custom_quant.apply(new_mem, quantization.global_ab)

    #     syn = new_syn
    #     mem = new_mem

    #     mem_rec.append(mem)
    #     spk_rec.append(out)


    #mem_rec = torch.stack(mem_rec,dim=1)
    #spk_rec = torch.stack(spk_rec,dim=1)


    #Readout layer
    h2 = einsum_linear.apply(spk_rec, spytorch_util.w2, scale2)


    g_e = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    v = torch.ones((batch_size,nb_outputs), device=device, dtype=dtype) * v_rest_e
    alpha = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    theta = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    v_thresh_e = torch.ones((batch_size,nb_outputs), device=device, dtype=dtype)*v_thresh_e_mult
    del_theta = torch.ones((batch_size,nb_outputs), device=device, dtype=dtype)*del_theta_mult

    out_rec = [v]
    
    for t in range(nb_steps-1):
        dge_dt = -g_e/tau_ge

        g_e, I_syn_E, dx_dt = g_e + time_step*dge_dt + h2[:,t]*inp_mult, (g_e*v_exc - g_e*v)*nS, -g_e/200e-3
        alpha = alpha + time_step*dx_dt
        # ferro
        dv_dt = (v_rest_e*alpha - v)/(.1*tau_v) + (I_syn_E/nS)/(1*tau_v)
        # ferro lif
        #dv_dt = (v_rest_e - v)/(1*tau_v) + (I_syn_E/nS)/(1*tau_v)
        v = v + time_step*dv_dt

        out_rec.append(v)

    out_rec = torch.stack(out_rec,dim=1)

    # flt = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    # out = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    # out_rec = [out]
    # for t in range(nb_steps):
    #     #import pdb; pdb.set_trace()
    #     new_flt = alpha1*flt +h2[:,t]
    #     #new_flt = custom_quant.apply(new_flt, quantization.global_ab)
    #     new_out = beta*out +flt
    #     #new_out = custom_quant.apply(new_out, quantization.global_ab)

    #     flt = new_flt 
    #     out = new_out 

    #     out_rec.append(out)

    # out_rec = torch.stack(out_rec,dim=1)


    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs


def train(x_data, y_data, lr, nb_epochs):
    params = [spytorch_util.w1,spytorch_util.w2]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9,0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    acc_hist = []
    acc_train = []
    for e in range(nb_epochs):
        local_loss = []
        accs = []

        
        for x_local, y_local in sparse_data_generator_DVS(x_data, y_data, batch_size, nb_steps, nb_inputs, shuffle = False, device=device):

            output,recs = run_snn(x_local.to_dense())
            _,spks=recs

            m,_=torch.max(output,1)



            _,am=torch.max(m,1)
            tmp = np.mean((y_local==am).detach().cpu().numpy())
            accs.append(tmp)



            log_p_y = log_softmax_fn(m)
            
            # Here we set up our regularizer loss
            # The strength paramters here are merely a guess and there should be ample room for improvement by
            # tuning these paramters.
            #reg_loss = 1e-5*torch.sum(spks) # L1 loss on total number of spikes
            #reg_loss += 1e-5*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron
            
            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) #+ reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())

        scheduler.step()
        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5f"%(e+1,mean_loss))
        acc_test = compute_classification_accuracy(x_test,y_test)
        #acc_train = compute_classification_accuracy(x_train,y_train)
        print("Test accuracy: %.3f"%(acc_test))
        print("Train accuracy: %.3f"%(np.mean(accs)))
        loss_hist.append(mean_loss)
        acc_hist.append(acc_test)
        acc_train.append(np.mean(accs))

        
    return loss_hist, acc_hist, acc_train
        

def compute_classification_accuracy(x_data, y_data):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    with torch.no_grad():
        for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs, shuffle=False):
            output,_ = run_snn(x_local.to_dense())
            m,_= torch.max(output,1) # max over time
            _,am=torch.max(m,1)      # argmax over output units
            tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
            accs.append(tmp)
    return np.mean(accs)



bit_string = str(quantization.global_wb) + str(quantization.global_ab) + str(quantization.global_gb) + str(quantization.global_eb)
print(bit_string)


spytorch_util.w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
scale1 = init_layer_weights(spytorch_util.w1, 28*28).to(device)

spytorch_util.w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
scale2 = init_layer_weights(spytorch_util.w2, 28*28).to(device)


loss_hist, test_acc, train_acc = train(x_train, y_train, lr = 1e-5, nb_epochs = 50)


results = {'bit_string': bit_string, 'test_acc': test_acc, 'test_loss': loss_hist, 'train_acc': train_acc ,'weight': [quant_w(spytorch_util.w1, scale1), quant_w(spytorch_util.w2, scale2)]}
date_string = time.strftime("%Y%m%d%H%M%S")


with open('results/snn_mnist_' + bit_string + '_' + str(inp_mult) + '_' + date_string + '.pkl', 'wb') as f:
    pickle.dump(results, f)



import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


plt.clf()
plt.plot(test_acc, label="test")
plt.plot(train_acc, label= "train")
plt.legend()
plt.title(bit_string + " " + str(inp_mult))
plt.savefig("./figures/ferro_mnist_"+date_string+".png")


