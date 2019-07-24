import os
import datetime

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn

import line_profiler


def conv_exp_kernel(inputs, time_step, tau, device):
    dtype = torch.float
    nb_hidden = inputs.shape[2]
    batch_size = inputs.shape[0]
    nb_steps = inputs.shape[1]
    alpha = float(torch.exp(torch.tensor([-time_step/tau], device= device, dtype=dtype)))

    u = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    rec_u = [u]
    
    for t in range(nb_steps-1):
        u = alpha*u + inputs[:,t,:]
        rec_u.append(u)

    rec_u = torch.stack(rec_u,dim=1)    
    return rec_u

def van_rossum(x, y, time_step, tau, device):
    tild_x = conv_exp_kernel(x, time_step, tau, device)
    tild_y = conv_exp_kernel(y, time_step, tau, device)
    return torch.sqrt(1/tau*torch.sum((tild_x - tild_y)**2))

