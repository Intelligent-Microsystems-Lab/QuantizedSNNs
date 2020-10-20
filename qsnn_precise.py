import  argparse, pickle

import torch
import torch.nn as nn
import torchvision
import numpy as np

import spytorch_util
import quantization

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

# Code is based on: https://github.com/fzenke/spytorch


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--input", type=str, default="./data/input_700_250_25.pkl", help='Input pickle')
parser.add_argument("--target", type=str, default="./data/smile95.pkl", help='Target pattern pickle')

parser.add_argument("--global_wb", type=int, default=2, help='Weight bitwidth')
parser.add_argument("--global_ab", type=int, default=8, help='Membrane potential, synapse state bitwidth')
parser.add_argument("--global_gb", type=int, default=8, help='Gradient bitwidth')
parser.add_argument("--global_eb", type=int, default=8, help='Error bitwidth')
parser.add_argument("--global_rb", type=int, default=16, help='Gradient RNG bitwidth')

parser.add_argument("--time_step", type=float, default=1e-3, help='Simulation time step size')
parser.add_argument("--nb_steps", type=float, default=250, help='Simulation steps')
parser.add_argument("--nb_epochs", type=float, default=10000, help='Simulation steps')

parser.add_argument("--tau_mem", type=float, default=10e-3, help='Time constant for membrane potential')
parser.add_argument("--tau_syn", type=float, default=5e-3, help='Time constant for synapse')
parser.add_argument("--tau_vr", type=float, default=5e-3, help='Time constant for Van Rossum distance')
parser.add_argument("--alpha", type=float, default=.75, help='Time constant for synapse')
parser.add_argument("--beta", type=float, default=.875, help='Time constant for Van Rossum distance')

parser.add_argument("--nb_inputs", type=int, default=700, help='Spatial input dimensions')
parser.add_argument("--nb_hidden", type=int, default=400, help='Spatial hidden dimensions')
parser.add_argument("--nb_outputs", type=int, default=250, help='Spatial output dimensions')

args = parser.parse_args()


quantization.global_wb = args.global_wb
quantization.global_ab = args.global_ab
quantization.global_gb = args.global_gb
quantization.global_eb = args.global_eb
quantization.global_rb = args.global_rb
stop_quant_level = 33

time_step = args.time_step
nb_steps  = args.nb_steps
tau_mem = args.tau_mem
tau_syn = args.tau_syn
tau_vr  = args.tau_vr

alpha   = args.alpha
beta    = args.beta

nb_inputs  = args.nb_inputs
nb_hidden  = args.nb_hidden
nb_outputs = args.nb_outputs

def conv_exp_kernel(inputs, time_step, tau, device):
    dtype = torch.float
    nb_hidden = inputs.shape[1]
    nb_steps = inputs.shape[0]

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
            w_quant = quantization.quant_w(weight, scale)
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
            quant_error = quantization.quant_err(grad_output)
        else:
            quant_error = grad_output

        if ctx.needs_input_grad[0]:
            # propagate quantized error
            grad_input = torch.einsum("bc,dc->bd", (quant_error, w_quant))

        if ctx.needs_input_grad[1]:
            if quantization.global_gb < stop_quant_level:
                grad_weight = quantization.quant_grad(torch.einsum("bc,bd->dc", (quant_error, input))).float()
            else:
                grad_weight = torch.einsum("bc,bd->dc", (quant_error, input))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class custom_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, b_level):
        if quantization.global_ab < stop_quant_level:
            output, clip_info = quantization.quant_act(input)
        else:
            output, clip_info = input, None
        ctx.save_for_backward(clip_info)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        clip_info = ctx.saved_tensors
        if quantization.global_eb < stop_quant_level:
            quant_error = quantization.quant_err(grad_output) * clip_info[0].float()
        else:
            quant_error = grad_output
        return quant_error, None


def run_snn(inputs):
    with torch.no_grad():
        spytorch_util.w1.data = quantization.clip(spytorch_util.w1.data, quantization.global_wb)
        spytorch_util.w2.data = quantization.clip(spytorch_util.w2.data, quantization.global_wb)


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


quantization.global_beta = quantization.step_d(quantization.global_wb)-.5
with open(args.input, 'rb') as f:
    x_train = pickle.load(f).t().to(device)


with open(args.target, 'rb') as f:
    y_train = torch.tensor(pickle.load(f)).to(device)
y_train = y_train.type(dtype)


bit_string = str(quantization.global_wb) + str(quantization.global_ab) + str(quantization.global_gb) + str(quantization.global_eb)

print("Start Training")
print(bit_string)

spytorch_util.w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
scale1 = quantization.init_layer_weights(spytorch_util.w1, 28*28).to(device)

spytorch_util.w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
scale2 = quantization.init_layer_weights(spytorch_util.w2, 28*28).to(device)


quantization.global_lr = .1
loss_hist, output = train(x_train, y_train, lr = 1, nb_epochs = args.nb_epochs)

bit_string = str(quantization.global_wb) + str(quantization.global_ab) + str(quantization.global_gb) + str(quantization.global_eb)

results = {'bit_string': bit_string ,'loss_hist': loss_hist, 'output': output.cpu()}

with open('results/snn_smile_precise_'+bit_string+'.pkl', 'wb') as f:
    pickle.dump(results, f)



