import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torchvision

from quantization import quant_act, init_layer_weights, SSE, to_cat, clip, quant_w
from spytorch_util import current2firing_time, sparse_data_generator, plot_voltage_traces, SuperSpike


# The coarse network structure is dicated by the Fashion MNIST dataset. 
nb_inputs  = 28*28
nb_hidden  = 100
nb_outputs = 10

time_step = 1e-3
nb_steps  = 100

batch_size = 256



dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")



class SuperSpike_Linear(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        #w_quant = quant_w(weight, scale)
        w_quant = weight

        h1 = torch.einsum("abc,cd->abd", (input, w_quant))
        syn = torch.zeros((batch_size,weight.shape[1]), device=device, dtype=dtype)
        mem = torch.zeros((batch_size,weight.shape[1]), device=device, dtype=dtype)

        mem_rec = [mem]
        spk_rec = [mem]
        mthr_rec = [mem]

        for t in range(nb_steps):
            mthr = mem-1.0
            #out = spike_fn(mthr)

            out = torch.zeros_like(mthr)
            out[mthr > 0] = 1.0

            rst = torch.zeros_like(mem)
            c   = (mthr > 0)
            rst[c] = torch.ones_like(mem)[c]

            new_syn = alpha*syn +h1[:,t]
            new_mem = beta*mem +syn -rst

            mem = new_mem
            syn = new_syn

            mthr_rec.append(mthr)
            mem_rec.append(mem)
            spk_rec.append(out)

        mthr_rec = torch.stack(mthr_rec,dim=1)
        mem_rec = torch.stack(mem_rec,dim=1)
        spk_rec = torch.stack(spk_rec,dim=1)



        ctx.save_for_backward(mthr_rec, w_quant)
        #out = torch.zeros_like(input)
        #out[input > 0] = 1.0
        return mem_rec, spk_rec

    @staticmethod
    def backward(ctx, grad_mem, grad_spk):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).

        There will be no gradient for mem, since its not used.
        """

        import pdb; pdb.set_trace()
        input, w_quant = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        quant_error = grad_spk

        # input 64, 512
        # error 64, 10
        # w     10, 512
        # grad_input 64, 512

        if ctx.needs_input_grad[0]:
            # propagate quantized error
            grad_input = torch.einsum("abc,cd->abd", (quant_error, w_quant.t()))
            #grad_input = quant_error.mm(w_quant)

        if ctx.needs_input_grad[1]:
            grad = quant_error/(SuperSpike.scale*torch.abs(input)+1.0)**2
            grad_weight = quant_error.t().mm(input)


            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


# Here we load the Dataset
train_dataset = torchvision.datasets.MNIST('../data', train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.MNIST('../data', train=False, transform=None, target_transform=None, download=True)

# Standardize data
# x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
x_train = np.array(train_dataset.train_data, dtype=np.float)
x_train = x_train.reshape(x_train.shape[0],-1)/255
# x_test = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
x_test = np.array(test_dataset.test_data, dtype=np.float)
x_test = x_test.reshape(x_test.shape[0],-1)/255

# y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
# y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)
y_train = np.array(train_dataset.train_labels, dtype=np.int)
y_test  = np.array(test_dataset.test_labels, dtype=np.int)

tau_mem = 10e-3
tau_syn = 5e-3

alpha   = float(np.exp(-time_step/tau_syn))
beta    = float(np.exp(-time_step/tau_mem))


weight_scale = 0.2

w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

print("init done")
    
# here we overwrite our naive spike function by the "SuperSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SuperSpike.apply


def run_snn(inputs):
    # h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    # syn = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    # mem = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)

    # mem_rec = [mem]
    # spk_rec = [mem]

    # # Compute hidden layer activity
    # for t in range(nb_steps):
    #     mthr = mem-1.0
    #     out = spike_fn(mthr)
    #     rst = torch.zeros_like(mem)
    #     c   = (mthr > 0)
    #     rst[c] = torch.ones_like(mem)[c]

    #     new_syn = alpha*syn +h1[:,t]
    #     new_mem = beta*mem +syn -rst

    #     mem = new_mem
    #     syn = new_syn

    #     mem_rec.append(mem)
    #     spk_rec.append(out)

    # mem_rec = torch.stack(mem_rec,dim=1)
    # spk_rec = torch.stack(spk_rec,dim=1)

    mem_rec1, spk_rec1 = SuperSpike_Linear.apply(inputs, w1)

    # Readout layer
    h2= torch.einsum("abc,cd->abd", (spk_rec1, w2))
    flt = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps):
        new_flt = alpha*flt +h2[:,t]
        new_out = beta*out +flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec,dim=1)
    other_recs = [mem_rec1, spk_rec1]
    return out_rec, other_recs


def train(x_data, y_data, lr=1e-3, nb_epochs=10):
    params = [w1,w2]
    optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs):
            output,recs = run_snn(x_local.to_dense())
            _,spks=recs
            m,_=torch.max(output,1)
            log_p_y = log_softmax_fn(m)
            
            # Here we set up our regularizer loss
            # The strength paramters here are merely a guess and there should be ample room for improvement by
            # tuning these paramters.
            reg_loss = 1e-5*torch.sum(spks) # L1 loss on total number of spikes
            reg_loss += 1e-5*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron
            
            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) + reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            import pdb; pdb.set_trace()
            local_loss.append(loss_val.item())
        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5f"%(e+1,mean_loss))
        loss_hist.append(mean_loss)
        
    return loss_hist
        

def compute_classification_accuracy(x_data, y_data):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs, shuffle=False):
        output,_ = run_snn(x_local.to_dense())
        m,_= torch.max(output,1) # max over time
        _,am=torch.max(m,1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
    return np.mean(accs)


loss_hist = train(x_train, y_train, lr=2e-4, nb_epochs=30)


print("Training accuracy: %.3f"%(compute_classification_accuracy(x_train,y_train)))
print("Test accuracy: %.3f"%(compute_classification_accuracy(x_test,y_test)))


#(Pdb) w1.grad.sum()
#tensor(0.2043, device='cuda:0')
