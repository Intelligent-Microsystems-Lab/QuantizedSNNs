import os
import time
import argparse
import re
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torchvision

import quantization
import spytorch_util
from quantization import init_layer_weights, clip, quant_w, quant_err, quant_grad
from spytorch_util import current2firing_time, sparse_data_generator_DVS, plot_voltage_traces, SuperSpike



ap = argparse.ArgumentParser()
ap.add_argument("-wb", "--wb", type = int, help = "weight bits")
ap.add_argument("-m", "--m", type = int, help="multiplier")
ap.add_argument("-rg", "--rg", type = float, help="reg")
ap.add_argument("-s1", "--s1", type = float, help="sum1")
ap.add_argument("-s2", "--s2", type = float, help="sum2")
args = vars(ap.parse_args())


quantization.global_wb = args['wb']
inp_mult = args['m']
reg_size = args['rg']
sum1v = args['s1']#*2.1
sum2v = args['s2']#*0.003


if quantization.global_wb == None:
    quantization.global_wb = 33
if inp_mult == None:
    inp_mult = 80 # 90 yielded high results for full

if sum1v == None:
    sum1v = 2.1
if sum2v == None:
    sum2v = 0.003



if reg_size == None:
    reg1 = 1e-03
    reg2 = reg1
else:
    reg1 = reg_size
    reg2 = reg_size



# Neuron Parameters
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



quantization.global_lr = 4e-4
batch_size = 128
nb_hidden  = 1500
nb_steps  =  150 # 100 previously, some good results with 150


#bernarbe tricks
threshold_saturation = del_theta_mult * 7 # the number 7 is the foundation of God's word ... lets hope

mult_eq = .12
class_method = "integrate"


nb_inputs  = 128*128
nb_outputs = 12
time_step = 1e-3 
dtype = torch.float
stop_quant_level = 32
quantization.global_gb = 33
quantization.global_eb = 33




# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")


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


print("init done")
    
# here we overwrite our naive spike function by the "SuperSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SuperSpike.apply


def run_snn(inputs, infer):

    with torch.no_grad():
        spytorch_util.w1.data = clip(spytorch_util.w1.data, quantization.global_wb)
        spytorch_util.w2.data = clip(spytorch_util.w2.data, quantization.global_wb)

    
    h1 = einsum_linear.apply(inputs, spytorch_util.w1*inp_mult, scale1)

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


        g_e, I_syn_E, dx_dt = g_e + time_step*dge_dt + h1[:,t], (g_e*v_exc - g_e*v)*nS, -g_e/200e-3
        alpha = alpha + time_step*dx_dt
        # ferro
        dv_dt = (v_rest_e*alpha - v)/(mult_eq*tau_v) + (I_syn_E/nS)/(1*tau_v) #.12
        # ferro lif
        #dv_dt = (v_rest_e - v)/(1*tau_v) + (I_syn_E/nS)/(1*tau_v)
        v = v + time_step*dv_dt

        out = spike_fn(v - (v_thresh_e+theta))
        c = (out==1.0)
        # threhshold increase, bernarbe trick 1
        theta[c] += del_theta[c] 

        # neuron threshold saturation, bernarbe trick 3
        theta[theta > threshold_saturation] = threshold_saturation

        # lateral inhibition, bernarbe trick 4

        alpha[c] = 1
        v[c] = v_reset_e[c]

        mem_rec.append(v)
        spk_rec.append(out)

    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)



    #Readout layer #infer is fine
    h2 = einsum_linear.apply(spk_rec, spytorch_util.w2*inp_mult, scale2)


    g_e = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    v = torch.ones((batch_size,nb_outputs), device=device, dtype=dtype) * v_rest_e
    alpha = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    theta = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    v_thresh_e = torch.ones((batch_size,nb_outputs), device=device, dtype=dtype)*v_thresh_e_mult
    del_theta = torch.ones((batch_size,nb_outputs), device=device, dtype=dtype)*del_theta_mult

    out_rec = [v]
    
    for t in range(nb_steps-1):
        dge_dt = -g_e/tau_ge

        g_e, I_syn_E, dx_dt = g_e + time_step*dge_dt + h2[:,t], (g_e*v_exc - g_e*v)*nS, -g_e/200e-3
        alpha = alpha + time_step*dx_dt
        # ferro
        dv_dt = (v_rest_e*alpha - v)/(mult_eq*tau_v) + (I_syn_E/nS)/(1*tau_v) #.12
        # ferro lif
        #dv_dt = (v_rest_e - v)/(1*tau_v) + (I_syn_E/nS)/(1*tau_v)
        v = v + time_step*dv_dt

        out_rec.append(v)

    out_rec = torch.stack(out_rec,dim=1)

    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs


def compute_classification_accuracy(x_data, y_data):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    with torch.no_grad():
        for x_local, y_local in sparse_data_generator_DVS(x_data, y_data, batch_size, nb_steps, nb_inputs, shuffle=True, time_step = time_step, device = device):
            output,_ = run_snn(x_local.to_dense(), True)

            if class_method == 'integrate':
                m = output.sum(axis = 1) # integrate
            else:
                m,_= torch.max(output,1) # max over time

            _,am=torch.max(m,1)      # argmax over output units

            tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
            accs.append(tmp)
    return np.mean(accs)


def train(x_data, y_data, lr, nb_epochs):
    params = [spytorch_util.w1,spytorch_util.w2]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9,0.999))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    acc_hist = []
    acc_train = []

    best = {'weights': [spytorch_util.w1, spytorch_util.w2], 'test_acc':0}
    for e in range(nb_epochs):
        local_loss = []
        accs = []

        
        for x_local, y_local in sparse_data_generator_DVS(x_data, y_data, batch_size, nb_steps, nb_inputs, shuffle = True, time_step = time_step, device = device):

            output,recs = run_snn(x_local.to_dense(), False)

            if class_method == 'integrate':
                m = output.sum(axis = 1) # integrate
            else:
                m,_=torch.max(output,1) # max val

            _,am=torch.max(m,1)
            tmp = np.mean((y_local==am).detach().cpu().numpy())
            accs.append(tmp)

            
            log_p_y = log_softmax_fn(m)

            # bernarbe trick 2 -> l2
            loss_val = loss_fn(log_p_y, y_local) + reg1 * ((spytorch_util.w1**2).sum(axis=0) - sum1v).abs().sum() + reg2 * ((spytorch_util.w2**2).sum(axis=0) - sum2v).abs().sum()
            # bernarbe trick 2 -> l1
            #loss_val = loss_fn(log_p_y, y_local) + reg1 * ((spytorch_util.w1.abs()).sum(axis=0) - 20).sum() + reg2 * ((spytorch_util.w2.abs()).sum(axis=0) - 12).sum()
            #+ (torch.sum(torch.abs(spytorch_util.w1)) + torch.sum(torch.abs(spytorch_util.w1)))*reg_size

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if torch.isnan(spytorch_util.w2.sum()) or  torch.isnan(spytorch_util.w1.sum()):
                break

            # normalize weights, bernarbe trick 2
            # with torch.no_grad():
            #     spytorch_util.w1.data = (spytorch_util.w1.data - spytorch_util.w1.data.min()) / (spytorch_util.w1.data - spytorch_util.w1.data.min()).sum() * first_sum
            #     spytorch_util.w2.data = (spytorch_util.w2.data - spytorch_util.w2.data.min()) / (spytorch_util.w2.data - spytorch_util.w2.data.min()).sum() * second_sum
            #     spytorch_util.w1.data = clip(spytorch_util.w1.data, quantization.global_wb)
            #     spytorch_util.w2.data = clip(spytorch_util.w2.data, quantization.global_wb)


            local_loss.append(loss_val.item())

        if torch.isnan(spytorch_util.w2.sum()) or  torch.isnan(spytorch_util.w1.sum()):
            break

        scheduler.step()
        mean_loss = np.mean(local_loss)
        acc_test = compute_classification_accuracy(x_test,y_test)
        print("Epoch %i: loss=%.5f"%(e+1,mean_loss))
        print("Test accuracy: %.3f"%(acc_test))
        print("Train accuracy: %.3f"%(np.mean(accs)))

        if best['test_acc'] < acc_test:
            best['test_acc'] = acc_test
            best['weights'] = [spytorch_util.w1, spytorch_util.w2]
        loss_hist.append(mean_loss)
        acc_hist.append(acc_test)
        acc_train.append(np.mean(accs))

        
    return loss_hist, acc_hist, acc_train, best
        




bit_string = str(quantization.global_wb)
para_dict = {'quantization.global_wb':quantization.global_wb, 'inp_mult':inp_mult, 'nb_hidden':nb_hidden, 'nb_steps':nb_steps, 'batch_size': batch_size, 'quantization.global_lr':quantization.global_lr, 'reg_size':reg1, 'mult_eq':mult_eq, 'class_method':class_method}
print(para_dict)

spytorch_util.w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
scale1 = init_layer_weights(spytorch_util.w1, 128*128).to(device)

spytorch_util.w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
scale2 = init_layer_weights(spytorch_util.w2, 2000).to(device)


loss_hist, test_acc, train_acc, best = train(x_train, y_train, lr = quantization.global_lr, nb_epochs = 35)


results = {'bit_string': bit_string, 'test_acc': test_acc, 'test_loss': loss_hist, 'train_acc': train_acc ,'weight': [spytorch_util.w1, spytorch_util.w2], 'best': best, 'para':para_dict, 'args': args}
date_string = time.strftime("%Y%m%d%H%M%S")


with open('results/snn_dvs_' + "_".join([re.sub('[^A-Za-z0-9.]+', '', x) for x in str(para_dict).split(" ")])+"_"+date_string + '.pkl', 'wb') as f:
    pickle.dump(results, f)



import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


plt.clf()
plt.plot(test_acc, label="test")
plt.plot(train_acc, label= "train")
plt.legend()
para_dict = {'quantization.global_wb':quantization.global_wb, 'inp_mult':inp_mult, 'reg_size':reg1, 'weight_sum': sum1v }
print(para_dict)
plt.title("_".join([re.sub('[^A-Za-z0-9.]+', '', x) for x in str(para_dict).split(" ")]))
plt.savefig("./figures/ferro_dvs_"+"_".join([re.sub('[^A-Za-z0-9.]+', '', x) for x in str(para_dict).split(" ")])+"_"+date_string+".png")


plt.clf()




# performance quant test

# test1 = pickle.load( open( "./results/snn_mnist_34_85_20191118071540.pkl", "rb" ) )



# quantization.global_wb = 2
# inp_mult = 250 # 90 yielded high results for full
# quantization.global_lr = 4e-4
# batch_size = 128
# nb_hidden  = 1050
# nb_steps  =  150 # 100 previously, some good results with 150
# reg_size = 0# 5e-5
# p_drop = 0



# nb_inputs  = 28*28
# nb_outputs = 10
# time_step = 1e-3 



# spytorch_util.w1 = test1['best']['weights'][0] 
# scale1 = 1

# spytorch_util.w2 = test1['best']['weights'][1] 
# scale2 = 1


# acc_test = compute_classification_accuracy(x_test,y_test)
# print(acc_test)

#print('test')
#for i in test1['test_acc']:
#    print(i)
#print("train")
#for i in test1['test_acc']:
#    print(i)

