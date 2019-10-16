import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# global quantization variables
global_wb = 8
global_ab = 8
global_gb = 8
global_eb = 8
global_rb = 16

global_beta = 1.5
global_lr = 8


# Quant Functions
def step_d(bits): 
    return 2.0 ** (bits - 1)

def shift(x):
    if x == 0:
        return 1
    return 2 ** torch.round(torch.log(x) / np.log(2))

def clip(x, bits):
    if bits == 1:
        delta = 0.
    else:
        delta = 1./step_d(bits)
    maxv = +1 - delta
    minv = -1 + delta
    return torch.clamp(x, float(minv), float(maxv))

def quant(x, bits):
    if bits == 1: # BNN
        return torch.sign(x)
    else:
        scale = step_d(bits)
        return torch.round(x * scale ) / scale

def quant_w(x, scale = 1):
    with torch.no_grad():
        y = quant(clip(x, global_wb) , global_wb)
        diff = (y - x)

    if scale <= 1.8:
        return x + diff
    return (x + diff)/scale

def quant_act(x):
    save_x = x
    x = clip(x, global_ab)
    diff_map = (save_x == x)
    with torch.no_grad():
        y = quant(x, global_ab)
        diff = y - x
    return x + diff, diff_map

def quant_grad(x):
    xmax = torch.max(torch.abs(x))
    x = x / shift(xmax)

    norm = quant(global_lr * x, global_rb)

    norm_sign = torch.sign(norm)
    norm_abs = torch.abs(norm)
    norm_int = torch.floor(norm_abs)
    norm_float = norm_abs - norm_int
    rand_float = torch.FloatTensor(x.shape).uniform_(0,1).to(x.device)
    norm = norm_sign.double() * ( norm_int.double() + 0.5 * (torch.sign(norm_float.double() - rand_float.double()) + 1) )

    return norm / step_d(global_gb)

def quant_err(x):
    alpha = shift(torch.max(torch.abs(x)))
    return quant(clip(x / alpha, global_eb), global_eb)

def init_layer_weights(weights_layer, shape, factor=1):
    fan_in = shape

    limit = torch.sqrt(torch.tensor([3*factor/fan_in]))
    Wm = global_beta/step_d(torch.tensor([float(global_wb)]))
    scale = 2 ** round(math.log(Wm / limit, 2.0))
    scale = scale if scale > 1 else 1.0
    limit = Wm if Wm > limit else limit

    torch.nn.init.uniform_(weights_layer, a = -float(limit), b = float(limit))
    return torch.tensor([float(scale)])

# sum of square errors
def SSE(y_true, y_pred):
    return 0.5 * torch.sum((y_true.double() - y_pred.double())**2)

def to_cat(inp_tensor, num_class, device):
    out_tensor = torch.zeros([inp_tensor.shape[0], num_class], device=device)
    out_tensor[torch.arange(inp_tensor.shape[0]).to(device), torch.tensor(inp_tensor, dtype = int, device=device)] = 1
    return out_tensor

# Inherit from Function
class clee_LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, scale, act, act_q, bias=None):
        # prep and save
        w_quant = quant_w(weight, scale)
        input = input.float()
        
        # compute output
        output = input.mm(w_quant.t())

        relu_mask = torch.ones(output.shape).to(output.device)
        clip_info = torch.ones(output.shape).to(output.device)

        # add relu and quant optionally
        if act:
            output = F.relu(output)
            relu_mask = (output != 0)
        if act_q:
            output, clip_info = quant_act(output)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, w_quant, bias, relu_mask, clip_info)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, w_quant, bias, relu_mask, clip_info = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        quant_error = quant_err(grad_output) * relu_mask.float() * clip_info.float()

        if ctx.needs_input_grad[0]:
            # propagate quantized error
            grad_input = quant_error.mm(w_quant)
        if ctx.needs_input_grad[1]:
            grad_weight = quant_grad(quant_error.t().mm(input)).float()
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None

# Inherit from Function
class clee_conv2d(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, scale, act=False, act_q=False, pool=False, bias=None):
        mpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        # prep and save
        w_quant = quant_w(weight, scale)
        input = input.float()
        
        # compute output
        output = F.conv2d(input, w_quant, bias=None, stride=1, padding=0, dilation=1, groups=1)

        relu_mask = torch.ones(output.shape).to(output.device)
        clip_info = torch.ones(output.shape).to(output.device)
        pool_indices = torch.ones(output.shape).to(output.device)
        size_pool = torch.tensor([0])

        # add pool, relu, quant optionally
        if pool:
            size_pool = output.shape
            output, pool_indices = mpool1(output)
        if act:
            output = F.relu(output)
            relu_mask = (output != 0)
        if act_q:
            output, clip_info = quant_act(output)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, w_quant, bias, torch.tensor([pool]), relu_mask, clip_info, pool_indices, torch.tensor(size_pool))
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        unpool1 = nn.MaxUnpool2d(2, stride=2, padding = 0)
        
        input, weight, w_quant, bias, pool, relu_mask, clip_info, pool_indices, size_pool = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_output = grad_output * relu_mask.float() * clip_info.float()
        if pool:
            grad_output = unpool1(grad_output, pool_indices, output_size = torch.Size(size_pool))

        quant_error = quant_err(grad_output) 

        if ctx.needs_input_grad[0]:
            # propagate quantized error
            grad_input = torch.nn.grad.conv2d_input(input.shape, w_quant, quant_error)
        if ctx.needs_input_grad[1]:
            grad_weight = quant_grad(torch.nn.grad.conv2d_weight(input, weight.shape, quant_error)).float()

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None