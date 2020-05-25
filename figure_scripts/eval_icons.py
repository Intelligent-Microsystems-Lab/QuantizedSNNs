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

import line_profiler


import glob, os
os.chdir("results/icons")



# base case vs not quantized
for file in glob.glob("*.pkl"):
    #print(file)

    # how to load
    with open(file, 'rb') as f:
       # The protocol version used is detected automatically, so we do not
       # have to specify it.
       data = pickle.load(f)

    #print(('args' in list(data)))
    #print(data['fname'].split("_")[1])
    # if ('args' in list(data)):
    #     print(data['fname'].split("_")[1])

    # if "WPQUEG812106610" == data['fname'].split("_")[1][:-1]:
    #     if ('args' in list(data)):
    #         if data['args'][10] == 0.001 and data['args'][11] == 0.001:
    #             print(data['fname'].split('_')[1][-1])
    #             print(max(data['acc']['test3']).item())
    #     else:
    #         print(data['fname'].split('_')[1][-1])
    #         print(max(data['acc']['test3']).item())
    #     #    print(file)

    #quantization.global_wb, quantization.global_qb, quantization.global_pb, quantization.global_rfb, quantization.global_sb, quantization.global_gb, quantization.global_eb, quantization.global_ub, quantization.global_ab, quantization.global_sig, quantization.global_rb,
    # if ('args' in list(data)):
    #     if data['args'][10] == 0.001 and data['args'][11] == 0.001:
    #         if data['args'][16] == 8 and data['args'][19] == 2 and data['args'][20] == 6 and data['args'][21] == 10 and  data['args'][22] == 6:
    #             print(data['args'][17])
    #             print(max(data['acc']['test3']).item())


    if ('args' in list(data)):
        #if data['args'][10] == 0.001 and data['args'][11] == 0.001:
        if data['args'][16] == 8 and data['args'][19] == 2 and data['args'][20] == 6 and data['args'][17] == 10 and  data['args'][18] == 12 and data['args'][21] == 10 and  data['args'][22] == 6:
            print(data['args'][10])
            print(data['args'][11])
            print(max(data['acc']['test3']).item())
            print(np.array(data['acc']['act_test2']).sum() + np.array(data['acc']['act_test1']).sum() + np.array(data['acc']['act_test3']).sum())



# base line
with open('d230c0ee-8509-11ea-b651-a0369ffa9370.pkl', 'rb') as f:
       # The protocol version used is detected automatically, so we do not
       # have to specify it.
       dataB = pickle.load(f)


# ful precision
with open('60b9f90a-7b6b-11ea-aa1a-a0369ffaa7c0.pkl', 'rb') as f:
       # The protocol version used is detected automatically, so we do not
       # have to specify it.
       dataN = pickle.load(f)


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')


plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='12')
fig, axes = plt.subplots(nrows=1, ncols=1) #

#plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)


axes.plot(dataN['acc']['test3'], label = 'FP (94.10%)',linewidth=2)
axes.plot(dataB['acc']['test3'], label = 'Quantized (91.67%)',linewidth=2)
axes.set_xlabel('Epochs')
axes.set_ylabel('Accuracy')
axes.legend()

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('../../figures/base_comp.png')
plt.close()



y = [0.881944477558136,
0.8993055820465088,
0.8958333134651184,
0.8958333134651184,
0.9166666865348816,
0.90625,
0.9027777910232544]

x = [2,3,4,5,6,7,8]

plt.clf()
plt.plot(x,y)
plt.xlabel('Readout Layer Weight Bits')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('../../figures/S_comp.png')
plt.close()



x = [6,7,8,9,10,11,12]

y = [0.756944477558136,
0.8055555820465088,
0.875,
0.9097222089767456,
0.913194477558136,
0.90625,
0.9201388955116272]

plt.clf()
plt.plot(x,y)
plt.xlabel('PQ Weight Bits')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('../../figures/PQ_comp.png')
plt.close()





import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

global_lr = 1
global_rb = 16

def step_d(bits): 
    return 2.0 ** (bits - 1)

def quant(x, bits):
    if bits == 1: # BNN
        return torch.sign(x)
    else:
        scale = step_d(bits)
        return torch.round(x * scale ) / scale

def shift(x):
    if x == 0:
        return 1
    return 2 ** torch.round(torch.log2(x))

def quant_grad(x):
    # those gonna be ternary, or we can tweak the lr
    xmax = torch.max(torch.abs(x))
    #import pdb; pdb.set_trace()
    norm = global_lr * x / shift(xmax)

    norm_sign = torch.sign(norm)
    norm_abs = torch.abs(norm)
    norm_int = torch.floor(norm_abs)
    norm_float = quant(norm_abs - norm_int, global_rb)
    rand_float = quant(torch.FloatTensor(x.shape).uniform_(0,1).to(x.device), global_rb)
    #norm = norm_sign.double() * ( norm_int.double() + 0.5 * (torch.sign(norm_float.double() - rand_float.double()) + 1) )
    zero_prevention_step = torch.sign(norm_float - rand_float)
    zero_prevention_step[zero_prevention_step == 0] = 1
    norm = norm_sign * ( norm_int + 0.5 * (zero_prevention_step + 1) )

    return norm / step_d(quant_level)



def quant_grad_trad(x):
    # those gonna be ternary, or we can tweak the lr
    xmax = torch.max(torch.abs(x))
    #import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    x = x/xmax
    norm = global_lr * x 

    norm_sign = torch.sign(norm)
    norm_abs = torch.abs(norm)
    norm_int = torch.floor(norm_abs)
    norm_float = quant(norm_abs - norm_int, global_rb)
    rand_float = quant(torch.FloatTensor(x.shape).uniform_(0,1).to(x.device), global_rb)
    zero_prevention_step = torch.sign(norm_float - rand_float)
    zero_prevention_step[zero_prevention_step == 0] = 1
    norm = norm_sign * ( norm_int + 0.5 * (zero_prevention_step + 1) )

    return norm / step_d(quant_level)



quant_level = 2
x = np.arange(start = -10, stop = 10, step = .001)
x1 = np.repeat(x[:, np.newaxis], 10000, 1)
xScatter = np.repeat(x[:, np.newaxis], 100, 1)
xScatter = np.repeat(xScatter[:,:, np.newaxis], 100, 2)
xScatterX = np.repeat(x[:, np.newaxis], 100, 1)

# #sigmoid
# grad = torch.sigmoid(torch.tensor(x))
# quant_grad = quant_grad_trad(grad)#*step_d(quant_level)
# mu = quant_grad.mean(axis = 1)
# std = quant_grad.std(axis = 1)

# plt.clf()
# plt.plot(x[:,0], mu, color='k')
# plt.fill_between(x[:,0], mu+std, mu-std, facecolor='k', alpha=0.3)
# plt.title("Surrogate Gradient PDF")
# plt.tight_layout()
# plt.savefig("something.png")


#derivative sigmoid
quant_level = 2
grad2 = (1-torch.sigmoid(torch.tensor(x1)))*torch.sigmoid(torch.tensor(x1))
quant_grad2 = quant_grad_trad(grad2.cpu())#*step_d(quant_level)
grad2Scatter = (1-torch.sigmoid(torch.tensor(xScatter)))*torch.sigmoid(torch.tensor(xScatter))
quant_grad2Scatter = quant_grad_trad(grad2Scatter.cpu())#*step_d(quant_level)
mu2 = quant_grad2.mean(axis = 1)
std2 = quant_grad2.std(axis = 1)
mu2Scatter = quant_grad2Scatter.mean(axis = 2)

quant_level = 4
grad8 = (1-torch.sigmoid(torch.tensor(x1)))*torch.sigmoid(torch.tensor(x1))
quant_grad8 = quant_grad_trad(grad8.cpu())#*step_d(quant_level)
grad8Scatter = (1-torch.sigmoid(torch.tensor(xScatter)))*torch.sigmoid(torch.tensor(xScatter))
quant_grad8Scatter = quant_grad_trad(grad8Scatter.cpu())#*step_d(quant_level)
mu8 = quant_grad8.mean(axis = 1)
std8 = quant_grad8.std(axis = 1)
mu8Scatter = quant_grad8Scatter.mean(axis = 2)

plt.clf()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')
plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='12')
fig, axes = plt.subplots(nrows=1, ncols=1) #
for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)


idx = np.random.choice(np.arange(len(xScatterX.flatten())) ,10000) 
axes.plot(x1[:,0], mu2, color='b')
#axes.fill_between(x[:,0], mu2+std2, mu2-std2, facecolor='b', alpha=0.3)
axes.scatter(xScatterX.flatten()[idx], mu2Scatter.flatten()[idx]  ,color='b',  s = .01) #alpha=0.7,


axes.plot(x1[:,0], mu8, color='g')
#axes.fill_between(x[:,0], mu8+std8, mu8-std8, facecolor='g', alpha=0.3)
axes.scatter(xScatterX.flatten()[idx], mu8Scatter.flatten()[idx]  ,color='g',  s = .1) #alpha=0.7,
plt.title("Surrogate Gradient PDF")
plt.tight_layout()
plt.savefig("something.png")









plt.clf()
plt.plot(x[:,0], grad[:,0], color='k', label='surrogate grad')
plt.plot(x[:,0], quant_grad[:,0], color='r', label='quant surrogate grad')
plt.legend()
plt.show()



#with variance

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
# how to load
with open("results/PQ.pkl", 'rb') as f:
  # The protocol version used is detected automatically, so we do not
  # have to specify it.
  data = pickle.load(f)

plt.clf()


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')


plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='12')
fig, axes = plt.subplots(nrows=1, ncols=1) #

#plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)
bins = 50
P_val = data['P']/data['P'].max()
Q_val = data['Q']/data['Q'].max()

P_val = P_val[P_val != 0.]
Q_val = Q_val[Q_val != 0.]

plt.hist(P_val, bins, alpha=0.5, label='P')
plt.hist(Q_val, bins, alpha=0.5, label='Q')
plt.legend(loc='lower center', bbox_to_anchor=(0.5,1.0), ncol=4, frameon=False)

plt.savefig("PQ_distr.png")


plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='12')
fig, axes = plt.subplots(nrows=1, ncols=1) #

#plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)
bins = 50
P_val = data['P']/data['P'].max()
Q_val = data['Q']/data['Q'].max()

P_val = P_val[P_val != 0.]
Q_val = Q_val[Q_val != 0.]

plt.hist(P_val, bins, alpha=0.5, label='P')
plt.hist(Q_val, bins, alpha=0.5, label='Q')
#plt.legend(loc='lower center', bbox_to_anchor=(0.5,1.0), ncol=4, frameon=False)
plt.yscale('log')
plt.savefig("PQ_distr_aux.png")




import numpy as np
import matplotlib.pyplot as plt

n = 100

#inS = [1]*n
inS = [1]*24 + [0]*(n-24)
#inS = [1]*100 + [0]*9900
#inS = ([1] + [0]*99) * 100
#inS = [1]*24 + [0]*(n-24)

P = 0
Q = 0
U = 0
R = 0
nU = 0
outS = 0

P1 = 0
Q1 = 0
U1 = 0
R1 = 0
nU1 = 0
outS1 = 0

P_hist = []
Q_hist = []
U_hist = []
R_hist = []
S_hist = []
outS_hist = []

P1_hist = []
Q1_hist = []
U1_hist = []
R1_hist = []
S1_hist = []
outS1_hist = []

weight_scale = 4e-5
weight =  float(1 / np.sqrt(1)) * 6 # / 250 * 1e-2) #*-1

perc_cap = 1

tau_mem = 20
tau_syn = 7.5
tau_ref = 1/.35

alpha = 1 - (1/tau_mem)#.95
beta =  1 - (1/tau_syn)#.86
gamma = 1 - (1/tau_ref)#.65

def quant01(x, bits = 8):
    scale = 2.0 ** (bits)
    return np.round(x * scale ) / scale

def quant11(x, bits = 8):
    signX = np.sign(x)
    scale = 2.0 ** (bits-1) -1
    temp = np.round(x*scale)/scale
    #temp[temp == 0 and signX == -1] = 1/scale
    if temp == 0 and signX == -1:
        return 1/scale
    return temp

q_scale = tau_syn/(1-beta)
p_scale = (tau_mem * q_scale*perc_cap)/(1-alpha)
r_scale = 1/(1-gamma)

inp_mult_q = 1/perc_cap * (1-beta)
inp_mult_p = 1/(perc_cap) * (1-alpha) #* (1/uncapped_max)

uncapped_max = inp_mult_q/(1-beta)

for t in range(n):


    Q, P, R = Q*beta + tau_syn * inS[t], P*alpha + tau_mem * Q,  gamma * R
    #Q1, P1, R1 = Q1*beta + .14 * inS[t], P1*alpha + .05 * Q1, 0.65 * R1
    

    if P > p_scale*perc_cap:
        P = p_scale*perc_cap
    if Q > q_scale*perc_cap:
        Q = q_scale*perc_cap

    U = P *weight*weight_scale - R
    S = float(U>=0)
    R += S

    Q1, P1, R1 = Q1*beta + inp_mult_q * inS[t], P1*alpha + inp_mult_p * Q1, gamma * R1
    if Q1 > 1:
        Q1 = 1
    if P1 > 1:
        P1 = 1
    #Q1 = quant01(Q1, 8)
    #P1 = quant01(P1, 8)
    R1 = quant01(R1, 8)

    U1 = P1 *weight*p_scale*perc_cap*weight_scale - R1*r_scale
    U1 = quant11(np.clip(U1,-8, 8)/8, 8)


    #U1_sign = np.sign(U1)
    #if U1_sign == -1:
    #    U1 = U1/(weight*p_scale*perc_cap*weight_scale)
    #if U1_sign == 1 or U1_sign == 0:
    #    U1 = U1/(weight*p_scale*perc_cap*weight_scale - r_scale)

    #U1 = quant01(U1, 8)
    #U1 *= U1_sign

    S1 = float(U1 >= 0)
    R1 += (1-gamma)*S1

    


    #import pdb; pdb.set_trace()
 
    #x = w + p_scale
    #p_scale = - w + x

    # cap inter spike interval
    # 

    #Q = Q/q_scale
    #P = P/p_scale



    

    P_hist.append(P)
    Q_hist.append(Q)
    U_hist.append(U)
    R_hist.append(R)
    S_hist.append(S)

    P1_hist.append(P1)
    Q1_hist.append(Q1)
    U1_hist.append(U1)
    R1_hist.append(R1)
    S1_hist.append(S1)



plt.clf()
fig, axs = plt.subplots(5, 3)
fig.set_size_inches(12.4, 8.8)
axs[0,0].plot(Q_hist, label="Q")
#axs[0,0].set_yscale('log')
axs[0,0].legend()
axs[0,1].plot(Q1_hist, label="Q1")
#axs[0,1].set_yscale('log')
axs[0,1].legend()
axs[0,2].plot([Q_hist[i] - Q1_hist[i]*q_scale*perc_cap for i in range(len(Q_hist))], label="diff Q")
axs[0,2].legend()
axs[1,0].plot(P_hist, label="P")
#axs[1,0].set_yscale('log')
axs[1,0].legend()
axs[1,1].plot(P1_hist, label="P1")
#axs[1,1].set_yscale('log')
axs[1,1].legend()
axs[1,2].plot([P_hist[i] - P1_hist[i]*p_scale*perc_cap for i in range(len(P_hist))], label="diff P")
axs[1,2].legend()
axs[2,0].plot(U_hist, label="U")
#axs[2,0].set_yscale('log')
axs[2,0].legend()
axs[2,1].plot(U1_hist, label="U1")
#axs[2,1].set_yscale('log')
axs[2,1].legend()
axs[2,2].plot([U_hist[i] - U1_hist[i] for i in range(len(U_hist))], label="diff U (not scaled)")
axs[2,2].legend()
axs[3,0].plot(R_hist, label="R")
axs[3,0].legend()
axs[3,1].plot(R1_hist, label="R1")
axs[3,1].legend()
axs[3,2].plot([R_hist[i] - R1_hist[i]*r_scale for i in range(len(R_hist))], label="diff R")
axs[3,2].legend()
axs[4,0].plot(S_hist, label="S")
axs[4,0].legend()
axs[4,1].plot(S1_hist, label="S1")
axs[4,1].legend()
axs[4,2].plot([S_hist[i] - S1_hist[i] for i in range(len(S_hist))], label="diff S")
axs[4,2].legend()
plt.tight_layout()
plt.savefig("PQ_testlo.png")


# from localQ import sparse_data_generator_Static, sparse_data_generator_DVSGesture, sparse_data_generator_DVSPoker, LIFConv2dLayer, prep_input, acc_comp, create_graph, hist_U_fun

# if torch.cuda.is_available():
#     device = torch.device("cuda")     
# else:
#     device = torch.device("cpu")
# dtype = torch.float

# ms = 1e-3
# T = 500*ms

# mean_isi = []
# median_isi = []
# min_isi = []
# max_seq = []
# for x_local, y_local in sparse_data_generator_DVSGesture(x_train, y_train, batch_size = 250, nb_steps = T / ms, shuffle = True, device = device):
    
#     test2 = x_local.reshape([-1, 500])
#     print("epoch")

#     for i in range(test2.shape[0]):
#         test3 = ((test2[i,:] != 0).nonzero())
#         if test3.nelement() != 0:
#             isi = torch.cat((test3[0], (test3[1:] - test3[:-1]).flatten()))
#             mean_isi.append(isi.float().mean())
#             median_isi.append(isi.float().median())
#             min_isi.append(isi.min())

#             cur = 1
#             temp = 1
#             for i in range(test3.size()[0]-1):
#                 if test3[i].item()+1 == test3[i+1].item():
#                     temp += 1
#                     if temp > cur:
#                         cur = temp
#                 else:
#                     temp = 1
#             max_seq.append(cur)



# plt.clf()
# fig, ax1 = plt.subplots()
# fig.set_size_inches(8.4, 4.8)
# ax1 = sns.distplot(layer1.weights.flatten().tolist())
# plt.title("Layer 1 Weights")
# plt.savefig('figures/weights1_'+str(uuid.uuid1())+'.png')
# plt.close()

# plt.clf()
# fig, ax1 = plt.subplots()
# fig.set_size_inches(8.4, 4.8)
# ax1 = sns.distplot(layer2.weights.flatten().tolist())
# plt.title("Layer 2 Weights")
# plt.savefig('figures/weights2_'+str(uuid.uuid1())+'.png')
# plt.close()

# plt.clf()
# fig, ax1 = plt.subplots()
# fig.set_size_inches(8.4, 4.8)
# ax1 = sns.distplot(layer3.weights.flatten().tolist())
# plt.title("Layer 3 Weights")
# plt.savefig('figures/weights3_'+str(uuid.uuid1())+'.png')
# plt.close()

# plt.clf()
# fig, ax1 = plt.subplots()
# fig.set_size_inches(8.4, 4.8)
# ax1 = sns.distplot([x.item() for x in median_isi])
# plt.title("median ISI")
# plt.savefig('figures/median_isi_'+str(uuid.uuid1())+'.png')
# plt.close()


# plt.clf()
# fig, ax1 = plt.subplots()
# fig.set_size_inches(8.4, 4.8)
# ax1 = sns.distplot([x.item() for x in min_isi])
# plt.title("min ISI")
# plt.savefig('figures/min_isi_'+str(uuid.uuid1())+'.png')
# plt.close()

# plt.clf()
# fig, ax1 = plt.subplots()
# fig.set_size_inches(8.4, 4.8)
# ax1 = sns.distplot(max_seq)
# plt.title("max seq")
# plt.savefig('figures/max_seq_'+str(uuid.uuid1())+'.png')
# plt.close()




# def quant11(x, bits = 8):
#     scale = 2.0 ** (bits-1) -1
#     return torch.round(x*scale)/scale

# test = torch.tensor(np.arange(-12, 12, 0.1))

# test2 = quant11(torch.clamp(test,-8, 8)/8, 4)
# test2.unique().shape




import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
import time
import math
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import uuid


import quantization
import localQ
from localQ import sparse_data_generator_Static, sparse_data_generator_DVSGesture, sparse_data_generator_DVSPoker, LIFConv2dLayer, prep_input, acc_comp, create_graph, DTNLIFConv2dLayer, create_graph2

import line_profiler

# No LR - DVS_WPQUEG8121066106_Inp0_LR1_Drop0.5_thrtensor([0.], device='cuda/0')_20200502_183001 - DVS_WPQUEG8121066106_Inp0_LR1_Drop0.5_thrtensor([0.], device='cuda/0')_20200502_183001.pkl
# Base - DVS_WPQUEG8121066106_Inp0_LR1_Drop0.5_thrtensor([0.], device='cuda/0')_20200502_183440 - DVS_WPQUEG8121066106_Inp0_LR1_Drop0.5_thrtensor([0.], device='cuda/0')_20200502_183440.pkl

# how to load
with open("results/NoneLong.pkl", 'rb') as f:
  # The protocol version used is detected automatically, so we do not
  # have to specify it.
  noneD = pickle.load(f)
  noneDD = [1-x.item() for x in noneD['acc']['test3']]


with open("results/BaseLong.pkl", 'rb') as f:
  # The protocol version used is detected automatically, so we do not
  # have to specify it.
  baseD = pickle.load(f)
  baseDD = [1-x.item() for x in baseD['acc']['test3']][:len(noneDD)]



with open("results/PQLong.pkl", 'rb') as f:
  # The protocol version used is detected automatically, so we do not
  # have to specify it.
  pqD = pickle.load(f)
  pqDD = [1-x.item() for x in pqD['acc']['test3']][:len(noneDD)]

with open("results/EGLong.pkl", 'rb') as f:
  # The protocol version used is detected automatically, so we do not
  # have to specify it.
  egD = pickle.load(f)
  egDD = [1-x.item() for x in egD['acc']['test3']][:len(noneDD)]

N = 3

noneDD = np.convolve(noneDD, np.ones((N,))/N, mode='valid')
baseDD = np.convolve(baseDD, np.ones((N,))/N, mode='valid')
pqDD = np.convolve(pqDD, np.ones((N,))/N, mode='valid')
egDD = np.convolve(egDD, np.ones((N,))/N, mode='valid')

plt.clf()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='16')
plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='16')
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.4,6.4)) #
for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)

axes.set_xlabel('Epochs')
axes.set_ylabel('Error')
axes.plot(noneDD, label='none')
axes.plot(baseDD, label='base')
axes.plot(pqDD, label='PQ')
axes.plot(egDD, label='EG')

axes.scatter(len(noneDD)-1, noneDD[-1], marker='d', color= '#1f77b4', s = 140, edgecolor='black', linewidth='2')
axes.text(len(noneDD)+1, noneDD[-1]+.001, r'(b)', fontsize=15,fontweight='bold')

axes.scatter(5, noneDD[5], marker='d', color= '#1f77b4', s = 140, edgecolor='black', linewidth='2')
axes.text(5, noneDD[5]+.015, r'(c)', fontsize=15,fontweight='bold')




axes.scatter(len(noneDD)-1, baseDD[-1], marker='d', color= '#ff7f0e', s = 140, edgecolor='black', linewidth='2')
axes.text(len(noneDD)-1, baseDD[-1]-0.04, r'(e)', fontsize=15,fontweight='bold')

axes.scatter(5, baseDD[5], marker='d', color= '#ff7f0e', s = 140, edgecolor='black', linewidth='2')
axes.text(5, baseDD[5]+.015, r'(f)', fontsize=15,fontweight='bold')




axes.scatter(len(noneDD)-1, egDD[-1], marker='d', color= '#d62728', s = 140, edgecolor='black', linewidth='2')
axes.text(len(noneDD)+1, egDD[-1]+.001, r'(i)', fontsize=15,fontweight='bold')

axes.scatter(5, egDD[5], marker='d', color= '#d62728', s = 140, edgecolor='black', linewidth='2')
axes.text(5, egDD[5]+.015, r'(h)', fontsize=15,fontweight='bold')




axes.scatter(len(noneDD)-1, pqDD[-1], marker='d', color= '#2ca02c', s = 140, edgecolor='black', linewidth='2',)
axes.text(len(noneDD)+1, pqDD[-1]+.001, r'(g)', fontsize=15,fontweight='bold')

axes.scatter(5, pqDD[5], marker='d', color= '#2ca02c', s = 140, edgecolor='black', linewidth='2',)
axes.text(5, pqDD[5]+.015, r'(d)', fontsize=15,fontweight='bold')


axes.legend(loc='lower center', bbox_to_anchor=(0.5,1.0), ncol=4, frameon=False)
#plt.title("Surrogate Gradient PDF")
#plt.legend()
#axes.set_aspect('equal', 'datalim')
plt.tight_layout()
plt.savefig("curves.pdf")







# NoLR
array([[ 1.       , -0.0176633],
       [-0.0176633,  1.       ]])

#Base
#all obs
array([[1.        , 0.35190249],
       [0.35190249, 1.        ]])

#first 60
[array([[1.        , 0.01041479],
        [0.01041479, 1.        ]]), 
array([[ 1.        , -0.11231557],
        [-0.11231557,  1.        ]]), 
array([[ 1.        , -0.19545752],
        [-0.19545752,  1.        ]]), 
array([[ 1.        , -0.16963043],
        [-0.16963043,  1.        ]]), 
array([[ 1.        , -0.12275829],
        [-0.12275829,  1.        ]])]
array([[1.       , 0.3426062],
       [0.3426062, 1.       ]])


0.35190249
0.01041479, -0.11231557, -0.19545752, -0.16963043, -0.12275829, 0.3426062


# print(test3)
# print(over_loss)
# print(test_acc_best_vali)

# WUPQR SASigEG Quantization: 8612102 666812 l1 0.001 l2 0.001 Inp 0 LR 1 Drop 0.5 Cap 0.75 thr 0.0
# Epoch Loss      Train1 Train2 Train3 Test1  Test2  Test3  | TrainT   TestT
# 01    1.989E+00 0.0543 0.6160 0.7128 0.0508 0.2458 0.3178 | 383.5978 89.8362


# 02    1.975E+00 0.0649 0.7266 0.8319 0.0636 0.5297 0.4746 | 421.2066 92.8690
# 03    2.057E+00 0.0755 0.7777 0.8734 0.0636 0.4915 0.4407 | 383.4044 90.4757
# 04    1.948E+00 0.0734 0.8043 0.9032 0.0636 0.4619 0.4534 | 385.4151 91.0888
# 05    1.855E+00 0.0766 0.8266 0.9053 0.0636 0.4195 0.4534 | 401.4738 91.3492



# how to load
with open("results3/Base.pkl", 'rb') as f:
  # The protocol version used is detected automatically, so we do not
  # have to specify it.
  dvsgest = pickle.load(f)

# with open("results/MNISTQ.pkl", 'rb') as f:
#   # The protocol version used is detected automatically, so we do not
#   # have to specify it.
#   mnist = pickle.load(f)

with open("results3/goodBase.pkl", 'rb') as f:
  # The protocol version used is detected automatically, so we do not
  # have to specify it.
  dvspoker = pickle.load(f)

with open("results3/Base.pkl", 'rb') as f:
  # The protocol version used is detected automatically, so we do not
  # have to specify it.
  dvspokerFF = pickle.load(f)

N = 5
pokerFF = 1-np.convolve([x.item() for x in dvspokerFF['acc']['test3']], np.ones((N,))/N, mode='valid')
N = 5
poker = 1-np.convolve([x.item() for x in dvspoker['acc']['test3']], np.ones((N,))/N, mode='valid')

valFF = 1- max(dvspokerFF['acc']['test3']).item()  

testBase = 1-dvspoker['evaled_test'].item()  
testFF = 1-dvspokerFF['evaled_test'].item()  

# gest = 1- np.mean( np.array([x.item() for x in dvsgest['acc']['test3']]).reshape(-1,2), axis=1)
# poker =  1-np.array([x.item() for x in poker['acc']['test3']])

plt.clf()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')
plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=1, ncols=1) #
for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)
#axes.get_xaxis().set_ticks([])
axes.set_xlabel('Epochs')
axes.set_ylabel('Error')
#axes.plot(noneDD, label='MNIST')
axes.plot(poker, label='DVS Poker', c='blue')
axes.plot(pokerFF, '--' ,c='orange')
axes.scatter(len(poker) + 5, testBase, c='blue', marker= "d", s = 100)
axes.scatter(len(poker) + 5, testFF, c='orange', marker= "d", s = 100)
#axes.plot(gest, label='DVS Gesture')
#axes.legend(loc='lower center', bbox_to_anchor=(0.5,1.0), ncol=4, frameon=False)


#plt.title("Surrogate Gradient PDF")
#plt.legend()
plt.tight_layout()
plt.savefig("data_set_poker.pdf")








with open("results3/NoneLong.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    noneD = pickle.load(f)
    noneTa1 = np.mean(noneD['acc']['act_train1'][217:222])
    noneTe1 = np.mean(noneD['acc']['act_test1'][-5:])
    noneW1 = np.mean(noneD['acc']['w1update'][-5:])
    noneTa2 = np.mean(noneD['acc']['act_train2'][-5:])
    noneTe2 = np.mean(noneD['acc']['act_test2'][-5:])
    noneW2 = np.mean(noneD['acc']['w2update'][-5:])
    noneTa3 = np.mean(noneD['acc']['act_train3'][-5:])
    noneTe3 = np.mean(noneD['acc']['act_test3'][-5:])
    noneW3 = np.mean(noneD['acc']['w3update'][-5:])

with open("results3/PQLong.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pqD = pickle.load(f)
    pqTa1 = np.mean(pqD['acc']['act_train1'][-5:])
    pqTe1 = np.mean(pqD['acc']['act_test1'][-5:])
    pqW1 = np.mean(pqD['acc']['w1update'][-5:])
    pqTa2 = np.mean(pqD['acc']['act_train2'][-5:])
    pqTe2 = np.mean(pqD['acc']['act_test2'][-5:])
    pqW2 = np.mean(pqD['acc']['w2update'][-5:])
    pqTa3 = np.mean(pqD['acc']['act_train3'][-5:])
    pqTe3 = np.mean(pqD['acc']['act_test3'][-5:])
    pqW3 = np.mean(pqD['acc']['w3update'][-5:])


with open("results3/EGLong.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    egD = pickle.load(f)
    egTa1 = np.mean(egD['acc']['act_train1'][-5:])
    egTe1 = np.mean(egD['acc']['act_test1'][-5:])
    egW1 = np.mean(egD['acc']['w1update'][-5:])
    egTa2 = np.mean(egD['acc']['act_train2'][-5:])
    egTe2 = np.mean(egD['acc']['act_test2'][-5:])
    egW2 = np.mean(egD['acc']['w2update'][-5:])
    egTa3 = np.mean(egD['acc']['act_train3'][-5:])
    egTe3 = np.mean(egD['acc']['act_test3'][-5:])
    egW3 = np.mean(egD['acc']['w3update'][-5:])





labels_list = []
acc_list = []
val_spikes = []

with open("results3/Base.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    baseD = pickle.load(f)
    baseTa1 = np.mean(baseD['acc']['act_train1'][217:222])
    baseTe1 = np.mean(baseD['acc']['act_test1'][217:222])
    baseW1 = np.mean(baseD['acc']['w1update'][217:222])
    baseTa2 = np.mean(baseD['acc']['act_train2'][217:222])
    baseTe2 = np.mean(baseD['acc']['act_test2'][217:222])
    baseW2 = np.mean(baseD['acc']['w2update'][217:222])
    baseTa3 = np.mean(baseD['acc']['act_train3'][217:222])
    baseTe3 = np.mean(baseD['acc']['act_test3'][217:222])
    baseW3 = np.mean(baseD['acc']['w3update'][217:222])

    labels_list.append('Base')
    acc_list.append(np.mean([x.item() for x in baseD['acc']['test3'][217:222]]))
    val_spikes.append(baseTe1 + baseTe2 + baseTe3)


with open("results3/PQ+2.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pqD = pickle.load(f)
    p2qTa1 = np.mean(pqD['acc']['act_train1'][217:222])
    p2qTe1 = np.mean(pqD['acc']['act_test1'][217:222])
    p2qW1 = np.mean(pqD['acc']['w1update'][217:222])
    p2qTa2 = np.mean(pqD['acc']['act_train2'][217:222])
    p2qTe2 = np.mean(pqD['acc']['act_test2'][217:222])
    p2qW2 = np.mean(pqD['acc']['w2update'][217:222])
    p2qTa3 = np.mean(pqD['acc']['act_train3'][217:222])
    p2qTe3 = np.mean(pqD['acc']['act_test3'][217:222])
    p2qW3 = np.mean(pqD['acc']['w3update'][217:222])

    labels_list.append('PQ+2')
    acc_list.append(np.mean([x.item() for x in pqD['acc']['test3'][217:222]]))
    val_spikes.append(p2qTe1 + p2qTe2 + p2qTe3)


with open("results3/PQ+1.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pqD = pickle.load(f)
    p1qTa1 = np.mean(pqD['acc']['act_train1'][217:222])
    p1qTe1 = np.mean(pqD['acc']['act_test1'][217:222])
    p1qW1 = np.mean(pqD['acc']['w1update'][217:222])
    p1qTa2 = np.mean(pqD['acc']['act_train2'][217:222])
    p1qTe2 = np.mean(pqD['acc']['act_test2'][217:222])
    p1qW2 = np.mean(pqD['acc']['w2update'][217:222])
    p1qTa3 = np.mean(pqD['acc']['act_train3'][217:222])
    p1qTe3 = np.mean(pqD['acc']['act_test3'][217:222])
    p1qW3 = np.mean(pqD['acc']['w3update'][217:222])

    labels_list.append('PQ+1')
    acc_list.append(np.mean([x.item() for x in pqD['acc']['test3'][217:222]]))
    val_spikes.append(p1qTe1 + p1qTe2 + p1qTe3)


with open("results3/PQ-1.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pqD = pickle.load(f)
    pm1qTa1 = np.mean(pqD['acc']['act_train1'][217:222])
    pm1qTe1 = np.mean(pqD['acc']['act_test1'][217:222])
    pm1qW1 = np.mean(pqD['acc']['w1update'][217:222])
    pm1qTa2 = np.mean(pqD['acc']['act_train2'][217:222])
    pm1qTe2 = np.mean(pqD['acc']['act_test2'][217:222])
    pm1qW2 = np.mean(pqD['acc']['w2update'][217:222])
    pm1qTa3 = np.mean(pqD['acc']['act_train3'][217:222])
    pm1qTe3 = np.mean(pqD['acc']['act_test3'][217:222])
    pm1qW3 = np.mean(pqD['acc']['w3update'][217:222])

    labels_list.append('PQ-1')
    acc_list.append(np.mean([x.item() for x in pqD['acc']['test3'][217:222]]))
    val_spikes.append(pm1qTe1 + pm1qTe2 + pm1qTe3)

with open("results3/PQ-2.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pqD = pickle.load(f)
    pm2qTa1 = np.mean(pqD['acc']['act_train1'][217:222])
    pm2qTe1 = np.mean(pqD['acc']['act_test1'][217:222])
    pm2qW1 = np.mean(pqD['acc']['w1update'][217:222])
    pm2qTa2 = np.mean(pqD['acc']['act_train2'][217:222])
    pm2qTe2 = np.mean(pqD['acc']['act_test2'][217:222])
    pm2qW2 = np.mean(pqD['acc']['w2update'][217:222])
    pm2qTa3 = np.mean(pqD['acc']['act_train3'][217:222])
    pm2qTe3 = np.mean(pqD['acc']['act_test3'][217:222])
    pm2qW3 = np.mean(pqD['acc']['w3update'][217:222])

    labels_list.append('PQ-2')
    acc_list.append(np.mean([x.item() for x in pqD['acc']['test3'][217:222]]))
    val_spikes.append(pm2qTe1 + pm2qTe2 + pm2qTe3)


with open("results3/PQ-3.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pqD = pickle.load(f)
    pm3qTa1 = np.mean(pqD['acc']['act_train1'][217:222])
    pm3qTe1 = np.mean(pqD['acc']['act_test1'][217:222])
    pm3qW1 = np.mean(pqD['acc']['w1update'][217:222])
    pm3qTa2 = np.mean(pqD['acc']['act_train2'][217:222])
    pm3qTe2 = np.mean(pqD['acc']['act_test2'][217:222])
    pm3qW2 = np.mean(pqD['acc']['w2update'][217:222])
    pm3qTa3 = np.mean(pqD['acc']['act_train3'][217:222])
    pm3qTe3 = np.mean(pqD['acc']['act_test3'][217:222])
    pm3qW3 = np.mean(pqD['acc']['w3update'][217:222])

    labels_list.append('PQ-3')
    acc_list.append(np.mean([x.item() for x in pqD['acc']['test3'][217:222]]))
    val_spikes.append(pm3qTe1 + pm3qTe2 + pm3qTe3)

with open("results3/PQ-4.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pqD = pickle.load(f)
    pm4qTa1 = np.mean(pqD['acc']['act_train1'][217:222])
    pm4qTe1 = np.mean(pqD['acc']['act_test1'][217:222])
    pm4qW1 = np.mean(pqD['acc']['w1update'][217:222])
    pm4qTa2 = np.mean(pqD['acc']['act_train2'][217:222])
    pm4qTe2 = np.mean(pqD['acc']['act_test2'][217:222])
    pm4qW2 = np.mean(pqD['acc']['w2update'][217:222])
    pm4qTa3 = np.mean(pqD['acc']['act_train3'][217:222])
    pm4qTe3 = np.mean(pqD['acc']['act_test3'][217:222])
    pm4qW3 = np.mean(pqD['acc']['w3update'][217:222])

    labels_list.append('PQ-4')
    acc_list.append(np.mean([x.item() for x in pqD['acc']['test3'][217:222]]))
    val_spikes.append(pm4qTe1 + pm4qTe2 + pm4qTe3)




with open("results3/EG+2.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    eqD = pickle.load(f)
    e2qTa1 = np.mean(eqD['acc']['act_train1'][217:222])
    e2qTe1 = np.mean(eqD['acc']['act_test1'][217:222])
    e2qW1 = np.mean(eqD['acc']['w1update'][217:222])
    e2qTa2 = np.mean(eqD['acc']['act_train2'][217:222])
    e2qTe2 = np.mean(eqD['acc']['act_test2'][217:222])
    e2qW2 = np.mean(eqD['acc']['w2update'][217:222])
    e2qTa3 = np.mean(eqD['acc']['act_train3'][217:222])
    e2qTe3 = np.mean(eqD['acc']['act_test3'][217:222])
    e2qW3 = np.mean(eqD['acc']['w3update'][217:222])

    labels_list.append('EG+2')
    acc_list.append(np.mean([x.item() for x in eqD['acc']['test3'][217:222]]))
    val_spikes.append(e2qTe1 + e2qTe2 + e2qTe3)


with open("results3/EG+1.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    eqD = pickle.load(f)
    e1qTa1 = np.mean(eqD['acc']['act_train1'][217:222])
    e1qTe1 = np.mean(eqD['acc']['act_test1'][217:222])
    e1qW1 = np.mean(eqD['acc']['w1update'][217:222])
    e1qTa2 = np.mean(eqD['acc']['act_train2'][217:222])
    e1qTe2 = np.mean(eqD['acc']['act_test2'][217:222])
    e1qW2 = np.mean(eqD['acc']['w2update'][217:222])
    e1qTa3 = np.mean(eqD['acc']['act_train3'][217:222])
    e1qTe3 = np.mean(eqD['acc']['act_test3'][217:222])
    e1qW3 = np.mean(eqD['acc']['w3update'][217:222])

    labels_list.append('EG+1')
    acc_list.append(np.mean([x.item() for x in eqD['acc']['test3'][217:222]]))
    val_spikes.append(e1qTe1 + e1qTe2 + e1qTe3)


with open("results3/EG-1.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    eqD = pickle.load(f)
    em1qTa1 = np.mean(eqD['acc']['act_train1'][217:222])
    em1qTe1 = np.mean(eqD['acc']['act_test1'][217:222])
    em1qW1 = np.mean(eqD['acc']['w1update'][217:222])
    em1qTa2 = np.mean(eqD['acc']['act_train2'][217:222])
    em1qTe2 = np.mean(eqD['acc']['act_test2'][217:222])
    em1qW2 = np.mean(eqD['acc']['w2update'][217:222])
    em1qTa3 = np.mean(eqD['acc']['act_train3'][217:222])
    em1qTe3 = np.mean(eqD['acc']['act_test3'][217:222])
    em1qW3 = np.mean(eqD['acc']['w3update'][217:222])

    labels_list.append('EG-1')
    acc_list.append(np.mean([x.item() for x in eqD['acc']['test3'][217:222]]))
    val_spikes.append(em1qTe1 + em1qTe2 + em1qTe3)

with open("results3/EG-2.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    eqD = pickle.load(f)
    em2qTa1 = np.mean(eqD['acc']['act_train1'][217:222])
    em2qTe1 = np.mean(eqD['acc']['act_test1'][217:222])
    em2qW1 = np.mean(eqD['acc']['w1update'][217:222])
    em2qTa2 = np.mean(eqD['acc']['act_train2'][217:222])
    em2qTe2 = np.mean(eqD['acc']['act_test2'][217:222])
    em2qW2 = np.mean(eqD['acc']['w2update'][217:222])
    em2qTa3 = np.mean(eqD['acc']['act_train3'][217:222])
    em2qTe3 = np.mean(eqD['acc']['act_test3'][217:222])
    em2qW3 = np.mean(eqD['acc']['w3update'][217:222])

    labels_list.append('EG-2')
    acc_list.append(np.mean([x.item() for x in eqD['acc']['test3'][217:222]]))
    val_spikes.append(em2qTe1 + em2qTe2 + em2qTe3)


with open("results3/EG-3.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    eqD = pickle.load(f)
    em3qTa1 = np.mean(eqD['acc']['act_train1'][217:222])
    em3qTe1 = np.mean(eqD['acc']['act_test1'][217:222])
    em3qW1 = np.mean(eqD['acc']['w1update'][217:222])
    em3qTa2 = np.mean(eqD['acc']['act_train2'][217:222])
    em3qTe2 = np.mean(eqD['acc']['act_test2'][217:222])
    em3qW2 = np.mean(eqD['acc']['w2update'][217:222])
    em3qTa3 = np.mean(eqD['acc']['act_train3'][217:222])
    em3qTe3 = np.mean(eqD['acc']['act_test3'][217:222])
    em3qW3 = np.mean(eqD['acc']['w3update'][217:222])

    labels_list.append('EG-3')
    acc_list.append(np.mean([x.item() for x in eqD['acc']['test3'][217:222]]))
    val_spikes.append(em3qTe1 + em3qTe2 + em3qTe3)

with open("results3/EG-4.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    eqD = pickle.load(f)
    em4qTa1 = np.mean(eqD['acc']['act_train1'][217:222])
    em4qTe1 = np.mean(eqD['acc']['act_test1'][217:222])
    em4qW1 = np.mean(eqD['acc']['w1update'][217:222])
    em4qTa2 = np.mean(eqD['acc']['act_train2'][217:222])
    em4qTe2 = np.mean(eqD['acc']['act_test2'][217:222])
    em4qW2 = np.mean(eqD['acc']['w2update'][217:222])
    em4qTa3 = np.mean(eqD['acc']['act_train3'][217:222])
    em4qTe3 = np.mean(eqD['acc']['act_test3'][217:222])
    em4qW3 = np.mean(eqD['acc']['w3update'][217:222])

    labels_list.append('EG-4')
    acc_list.append(np.mean([x.item() for x in eqD['acc']['test3'][217:222]]))
    val_spikes.append(em4qTe1 + em4qTe2 + em4qTe3)

with open("results3/BaseFF.pkl", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    baseD = pickle.load(f)
    baseTa1 = np.mean(baseD['acc']['act_train1'][217:222])
    baseTe1 = np.mean(baseD['acc']['act_test1'][217:222])
    baseW1 = np.mean(baseD['acc']['w1update'][217:222])
    baseTa2 = np.mean(baseD['acc']['act_train2'][217:222])
    baseTe2 = np.mean(baseD['acc']['act_test2'][217:222])
    baseW2 = np.mean(baseD['acc']['w2update'][217:222])
    baseTa3 = np.mean(baseD['acc']['act_train3'][217:222])
    baseTe3 = np.mean(baseD['acc']['act_test3'][217:222])
    baseW3 = np.mean(baseD['acc']['w3update'][217:222])

    #labels_list.append('Base')
    #acc_list.append(np.mean([x.item() for x in baseD['acc']['test3'][217:222]]))
    #val_spikes.append(baseTe1 + baseTe2 + baseTe3)
    FFB1 = np.mean([x.item() for x in baseD['acc']['test3'][217:222]])
    FFB2 = baseTe1 + baseTe2 + baseTe3


#train act

# set height of bar
PQbars1Ta = [pm4qTa1, pm3qTa1, pm2qTa1, pm1qTa1, baseTa1, p1qTa1, p2qTa1]
PQbars2Ta = [pm4qTa2, pm3qTa2, pm2qTa2, pm1qTa2, baseTa2, p1qTa2, p2qTa2]
PQbars3Ta = [pm4qTa3, pm3qTa3, pm2qTa3, pm1qTa3, baseTa3, p1qTa3, p2qTa3]

PQbars1Te = [pm4qTe1, pm3qTe1, pm2qTe1, pm1qTe1, baseTe1, p1qTe1, p2qTe1]
PQbars2Te = [pm4qTe2, pm3qTe2, pm2qTe2, pm1qTe2, baseTe2, p1qTe2, p2qTe2]
PQbars3Te = [pm4qTe3, pm3qTe3, pm2qTe3, pm1qTe3, baseTe3, p1qTe3, p2qTe3]

PQbars1W = [pm4qW1, pm3qW1, pm2qW1, pm1qW1, baseW1, p1qW1, p2qW1]
PQbars2W = [pm4qW2, pm3qW2, pm2qW2, pm1qW2, baseW2, p1qW2, p2qW2]
PQbars3W = [pm4qW3, pm3qW3, pm2qW3, pm1qW3, baseW3, p1qW3, p2qW3]




EQbars1Ta = [em4qTa1, em3qTa1, em2qTa1, em1qTa1, baseTa1, e1qTa1, e2qTa1]
EQbars2Ta = [em4qTa2, em3qTa2, em2qTa2, em1qTa2, baseTa2, e1qTa2, e2qTa2]
EQbars3Ta = [em4qTa3, em3qTa3, em2qTa3, em1qTa3, baseTa3, e1qTa3, e2qTa3]

EQbars1Te = [em4qTe1, em3qTe1, em2qTe1, em1qTe1, baseTe1, e1qTe1, e2qTe1]
EQbars2Te = [em4qTe2, em3qTe2, em2qTe2, em1qTe2, baseTe2, e1qTe2, e2qTe2]
EQbars3Te = [em4qTe3, em3qTe3, em2qTe3, em1qTe3, baseTe3, e1qTe3, e2qTe3]

EQbars1W = [em4qW1, em3qW1, em2qW1, em1qW1, baseW1, e1qW1, e2qW1]
EQbars2W = [em4qW2, em3qW2, em2qW2, em1qW2, baseW2, e1qW2, e2qW2]
EQbars3W = [em4qW3, em3qW3, em2qW3, em1qW3, baseW3, e1qW3, e2qW3]



bars1Ta = [noneTa1, baseTa1, pqTa1, egTa1]
bars2Ta = [noneTa2, baseTa2, pqTa2, egTa2]
bars3Ta = [noneTa3, baseTa3, pqTa3, egTa3]

bars1Te = [noneTe1, baseTe1, pqTe1, egTe1]
bars2Te = [noneTe2, baseTe2, pqTe2, egTe2]
bars3Te = [noneTe3, baseTe3, pqTe3, egTe3]

bars1W = [noneW1, baseW1, pqW1, egW1]
bars2W = [noneW2, baseW2, pqW2, egW2]
bars3W = [noneW3, baseW3, pqW3, egW3]
     
def barplot(bars1, bars2, bars3, name,title_str, labels_list):
    barWidth = 0.25
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]



    plt.clf()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rc('font', size='14')
    plt.clf()
    plt.rc('font', family='sans-serif')
    plt.rc('font', weight='bold')
    plt.rc('font', size='14')
    fig, axes = plt.subplots(nrows=1, ncols=1) #
    for axis in ['bottom','left']:
        axes.spines[axis].set_linewidth(2)
    for axis in ['top','right']:
        axes.spines[axis].set_linewidth(0)
    axes.xaxis.set_tick_params(width=2)
    axes.yaxis.set_tick_params(width=2)
     
    # Make the plot
    axes.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Layer 1')
    axes.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Layer 2')
    axes.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Layer 3')
     
    # Add xticks on the middle of the group bars
    #axes.set_xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], labels_list)
    plt.title(title_str)
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    #plt.title("Surrogate Gradient PDF")
    #plt.legend()
    plt.tight_layout()
    plt.savefig(name)


barplot(bars1Ta, bars2Ta, bars3Ta, "barsActTa.pdf", "Histogram Spikes Training",['None', 'Base', 'PQ', 'EG'])
barplot(bars1Te, bars2Te, bars3Te, "barsActTe.pdf", "Histogram Spikes Validation",['None', 'Base', 'PQ', 'EG'])
barplot(bars1W, bars2W, bars3W, "barsActW.pdf",  "Histogram Weight Updates",['None', 'Base', 'PQ', 'EG'])


barplot(PQbars1Ta, PQbars2Ta, PQbars3Ta, "PQbarsActTa.pdf", "Histogram Spikes Training",['PQ-4', 'PQ-3', 'PQ-2', 'PQ-1', 'Base', 'PQ+1', 'PQ+2'])
barplot(PQbars1Te, PQbars2Te, PQbars3Te, "PQbarsActTe.pdf", "Histogram Spikes Validation",['PQ-4', 'PQ-3', 'PQ-2', 'PQ-1', 'Base', 'PQ+1', 'PQ+2'])
barplot(PQbars1W, PQbars2W, PQbars3W, "PQbarsActW.pdf",  "Histogram Weight Updates",['PQ-4', 'PQ-3', 'PQ-2', 'PQ-1', 'Base', 'PQ+1', 'PQ+2'])


barplot(EQbars1Ta, EQbars2Ta, EQbars3Ta, "EGbarsActTa.pdf", "Histogram Spikes Training",['EG-4', 'EG-3', 'EG-2', 'EG-1', 'Base', 'EG+1', 'EG+2'])
barplot(EQbars1Te, EQbars2Te, EQbars3Te, "EGbarsActTe.pdf", "Histogram Spikes Validation",['EG-4', 'EG-3', 'EG-2', 'EG-1', 'Base', 'EG+1', 'EG+2'])
barplot(EQbars1W, EQbars2W, EQbars3W, "EgbarsActW.pdf",  "Histogram Weight Updates",['EG-4', 'EG-3', 'EG-2', 'EG-1', 'Base', 'EG+1', 'EG+2'])




plt.clf()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')
plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=1, ncols=1) #
for axis in ['bottom','left']:
    axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
    axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)
 

color_bla1 = np.array([[255*1, 0, 0], [255*.85, 0, 0], [255*.7, 0, 0],[255*.55, 0, 0], [255*.4, 0, 0], [255*.25, 0, 0]])
color_bla2 = np.array([[0, 255*1, 0], [0, 255*.85, 0], [0, 255*.7, 0],[0, 255*.55, 0], [0, 255*.4, 0], [0, 255*.25, 0 ]])

axes.scatter(acc_list[1:7],val_spikes[1:7], label = "PQ", marker = 's', c=color_bla1/255.0, s=100)
axes.scatter(acc_list[7:],val_spikes[7:], label = "EG", marker = 'o', c=color_bla2/255.0, s=100)
axes.scatter(acc_list[0],val_spikes[0], label = "base", marker = '*', s=100)
axes.scatter(FFB1,FFB2, label = "none", marker = 'd', s=100)
# Add xticks on the middle of the group bars
axes.set_xlabel('Accuracy', fontweight='bold')
axes.set_ylabel('# Spikes', fontweight='bold')
axes.legend(loc='upper center', bbox_to_anchor=(0.5, -.15), ncol=4, frameon=False)
#axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
#plt.title("Spike Activity Accuracy Trade-off")
#plt.legend()
plt.tight_layout()
plt.savefig("Scatter.pdf")









all_fnames = ["PQCap"+str(x)+".pkl" for x in [25, 50, 75, 80, 90, 100]]


for cur_fname in all_fnames:
    with open("results3/"+cur_fname, 'rb') as f:
        eqD = pickle.load(f)
        temp = np.array([x.item() for x in eqD['acc']['test3'][-40:]])
        print(cur_fname+" ", end='')
        print(eqD['evaled_test'], end=' ')
        print("{0:.4f}".format(temp.mean()), end=' ')
        print("{0:.4f}".format(temp.std()))







diffs = [-4, -3, -2, -1, 1, 2]
all_fnames = ["results3/S"+("+" if x > 0 else "")+str(x)+".pkl" for x in diffs] + ["results3/PQ"+("+" if x > 0 else "")+str(x)+".pkl" for x in diffs] + ["results3/EG"+("+" if x > 0 else "")+str(x)+".pkl" for x in diffs]

best_test_pos = []
test_perf = []
early1 = []
early2 = []
late = []


for cur_fname in all_fnames:
    with open(cur_fname, 'rb') as f:
        eqD = pickle.load(f)
        import pdb; pdb.set_trace()
        print(cur_fname, end='')
        print(eqD['evaled_test'])
        test3 = [x.item() for x in eqD['acc']['test3']]
        best_test_pos.append(np.argmax(test3))
        early1.append(np.mean(test3[45:50]))
        early2.append(np.mean(test3[95:100]))
        late.append(np.mean(test3[315:]))


with open("results3/Base.pkl", 'rb') as f:
    eqD = pickle.load(f)
    print(eqD['evaled_test'])

from os import listdir
from os.path import isfile, join
import os

mypath = "results3"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()

pq = []
p50 = []
best_vali = []
tyyy = []

for curf in onlyfiles:
    print(curf)
    with open(mypath + "/" + curf, 'rb') as f:
        eqD = pickle.load(f)
    print(len(eqD['acc']['test3']))
    #print(max(eqD['acc']['test3']))
    #import pdb; pdb.set_trace()

os.remove(mypath + "/" + "S-4_35558da8-96b2-11ea-9e31-acde48001122.pkl")

    if 'PQCap100' in curf:
        with open(mypath + "/" + curf, 'rb') as f:
            eqD = pickle.load(f)
            if eqD['args'][3] == 320:
                print("DVS Gesture")
            if eqD['args'][3] == 100:
                if eqD['layer1'][-2].shape[1] == 32:
                    print("Poker")
                else:
                    print("MNIST")
            print("PQ_cap {0}".format(eqD['args'][6]))
            print(eqD['fname'].split('_')[1])
            print("S 6: "+eqD['fname'].split('_')[1][-1])
            print("PQ 1210: "+eqD['fname'].split('_')[1][7:11])
            print("EG 106: "+eqD['fname'].split('_')[1][8:-2])
            #if eqD['args'][-12] is None:
            #    print("Floating Point")
            #else:
            #    print("PQ diff {0}".format(eqD['args'][-12]-12))
            #    print("S diff {0}".format(eqD['args'][-10]-6))
            #    print("EG diff {0}".format(eqD['args'][-9]-10))
            import pdb; pdb.set_trace()

        


#PQCap100
#GestureFF
#MNISTBase
os.rename(mypath + "/" + curf,mypath + "/" + "EG+1_"+str(uuid.uuid1())+".pkl") 


mypath = "/afs/crc.nd.edu/user/c/cschaef6/PQ_test"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

pq = []
p50 = []
best_vali = []
tyyy = []

for curf in onlyfiles:
    print(curf)
    with open(mypath + "/" + curf, 'rb') as f:
        eqD = pickle.load(f)
        auxV = [x.item() for x in eqD['acc']['test3']]
        pq.append(eqD['args'][6])
        p50.append(np.mean(auxV[35:40]) )
        tyyy.append(max(eqD['acc']['test3']).item())
        best_vali.append(eqD['evaled_test'].item() )


plt.clf()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')
plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=1, ncols=1) #
for axis in ['bottom','left']:
    axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
    axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)
 
axes.scatter(pq, best_vali, label='Test')
axes.scatter(pq, tyyy, label='Vali')
# Add xticks on the middle of the group bars
axes.set_xlabel('PQ cap', fontweight='bold')
axes.set_ylabel('Test Acc', fontweight='bold')
axes.legend(loc='upper center', bbox_to_anchor=(0.5, -.15), ncol=4, frameon=False)
#axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
plt.title("PQ Cap best ")
#plt.legend()
plt.tight_layout()
plt.savefig("pq_cap.pdf")



W = 8 #32
P = 12 #32
Q = 10 #32
R = 2 #32
U = 6 #32
S = 6 #32
A = 6 #32
G = 10#32
E = 6#32

l1 = 64*2*7*7 * W + 2*32*32 * P + 2*32*32 * Q + 1440*R + 1440*U + 2*11*1440*S + 11*A + 64*2*7*7 *G + (11+1440)*E
l2 = 128*64*7*7 * W + 64*15*15 * P + 64*15*15 * Q + 21632*R + 21632*U + 2*11*21632*S + 11*A + 128*64*7*7 * G + (11 + 21632) * E 
l3 = 128*128*7*7 * W + 128*13*13 * P + 128*13*13 * Q + 3200*R + 3200*U + 2*11*3200*S + 11*A +128*128*7*7 * G + (11 + 3200) * E

l1+l2+l3


#32 bits
100928576


#base 
26462796



W   Q   P   R   S   U/A/Sig E   G
8   10  12  2   6   6   6   10





