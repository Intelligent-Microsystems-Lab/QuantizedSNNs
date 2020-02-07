import torch
import torch.nn as nn
import torchvision
import numpy as np
import pickle
import time

import matplotlib.pyplot as plt


def spike_trains(rates, T, batches = 1):
    spike_data = torch.zeros(batches, len(rates),T)
    for j in range(batches):
        for i in range(len(rates)):
            spike_times = torch.cumsum(torch.distributions.poisson.Poisson(int(rates[i])).sample([int(T+rates[i])]), dim = 0)
            spike_times = spike_times[int(rates[i]):]
            spike_times = spike_times[spike_times < T]
            spike_data[j, i, :].index_fill_(0, spike_times.type(torch.LongTensor), 1)
    return spike_data

def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    """ Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

    Args:
    x -- The "current" values

    Keyword args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value 
    tmax -- The maximum time returned 
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Time to first spike for each "current" x
    """
    idx = x<thr
    x = np.clip(x, thr+epsilon, 1e9)
    T = tau*np.log(x / (x - thr))
    T[idx] = tmax
    return T

def aux_plot_i_u_s(inputs, rec_u, rec_s, batches, filename = ''):
    plt.clf()
    figure, axes = plt.subplots(nrows=3, ncols=batches)

    if batches == 1:
        i = 0
        axes[0].set_ylabel("Input Spikes")
        axes[1].set_ylabel("Neurons U(t)")
        axes[2].set_ylabel("Neurons S(t)")
        axes[0].set_title("Batch #"+str(i))
        axes[0].plot(inputs[i,:,:].nonzero()[:,1].cpu(), inputs[i,:,:].nonzero()[:,0].cpu(), 'k|')
        axes[0].set_yticklabels([])
        axes[0].set_xticklabels([])
        axes[0].set_xlim([0,len(rec_u[i,0,:])])
        for j in range(rec_u.shape[1]):
            axes[1].plot(rec_u[i,j,:].cpu()+j*5)
        axes[1].set_yticklabels([])
        axes[1].set_xticklabels([])
        axes[2].plot(rec_s[i,:,:].nonzero()[:,1].cpu(), rec_s[i,:,:].nonzero()[:,0].cpu(), 'k|')
        axes[2].set_yticklabels([])
        axes[2].set_xlim([0,len(rec_u[i,0,:])])
        axes[2].set_xlabel('Time (t)')

        plt.tight_layout()
        if filename == '':
            plt.show()
        else:
            plt.savefig(filename)


    elif batches > 1:
        axes[0, 0].set_ylabel("Input Spikes")
        axes[1, 0].set_ylabel("Neurons U(t)")
        axes[2, 0].set_ylabel("Neurons S(t)")
        for i in range(batches):
            axes[0, i].set_title("Batch #"+str(i))
            axes[0, i].plot(inputs[i,:,:].nonzero()[:,1].cpu(), inputs[i,:,:].nonzero()[:,0].cpu(), 'k|')
            axes[0, i].set_yticklabels([])
            axes[0, i].set_xticklabels([])
            axes[0, i].set_xlim([0,len(rec_u[i,0,:])])
            for j in range(rec_u.shape[1]):
                axes[1, i].plot(rec_u[i,j,:].cpu()+j*5)
            axes[1, i].set_yticklabels([])
            axes[1, i].set_xticklabels([])
            axes[2, i].plot(rec_s[i,:,:].nonzero()[:,1].cpu(), rec_s[i,:,:].nonzero()[:,0].cpu(), 'k|')
            axes[2, i].set_yticklabels([])
            axes[2, i].set_xlim([0,len(rec_u[i,0,:])])
            axes[2, i].set_xlabel('Time (t)')

        plt.tight_layout()
        if filename == '':
            plt.show()
        else:
            plt.savefig(filename)
    else:
        print('Bad number of batches to display')



def sparse_data_generator(X, y, batch_size, nb_steps, shuffle=True, time_step=1e-3, device=torch.device("cpu")):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors. 

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """


    labels_ = np.array(y, dtype=np.int)
    number_of_batches = len(X)//batch_size
    sample_index = np.arange(len(X))
    nb_units = X.shape[1]

    # compute discrete firing times
    tau_eff = 20e-3/time_step
    firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=np.int)
    unit_numbers = np.arange(nb_units)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            c = firing_times[idx]<nb_steps
            times, units = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
    
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index],device=device)

        try:
            yield X_batch.to(device=device).to_dense(), y_batch.to(device=device)
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
        return grad_input

superspike = SuperSpike().apply


class LIFDenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size, bias=True, alpha = .9, beta=.85, firing_threshold = 1, device=torch.device("cpu"), dtype = torch.float):
        super(LIFDenseLayer, self).__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.firing_threshold = firing_threshold
        self.batch_size = batch_size

        self.weights = torch.empty((in_channels, out_channels),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.uniform_(self.weights, a = -.3, b = .3)

        self.bias = torch.empty((out_channels),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.uniform_(self.bias, a = -.01, b = .01)

        self.P = torch.zeros(self.batch_size, self.in_channels).detach().to(device)
        self.Q = torch.zeros(self.batch_size, self.in_channels).detach().to(device)
        self.R = torch.zeros(self.batch_size, self.out_channels).detach().to(device)
        self.S = torch.zeros(self.batch_size, self.out_channels).detach().to(device)
        self.U = torch.zeros(self.batch_size, self.out_channels).detach().to(device)
    
    
    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.alpha * self.R - self.S, self.beta * self.Q + input_t
        self.U = torch.einsum("ab,bc->ac", (self.P, self.weights)) + self.bias + self.R
        self.S = (self.U>self.firing_threshold).float()

        return self.S



class LinearFAFunction(torch.autograd.Function):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_fa)
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias

class FALinear(nn.Module):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    def __init__(self, input_features, output_features, bias=True):
        super(FALinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # weight and bias for forward pass
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # fixed random weight and bias for FA backward pass does not need gradient
        self.weight_fa = torch.nn.Parameter(torch.FloatTensor(output_features, input_features), requires_grad=False)
        self.reset_lc_parameters(self)

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)
    
    @staticmethod
    def reset_lc_parameters(layer):
        #Random initialization with 50% variation is done here
        if hasattr(layer, 'weight_fa'):
            layer.weight_fa.data.normal_(1, .5)
            layer.weight_fa.data[layer.weight_fa.data<0] = 0
            layer.weight_fa.data[:] *= layer.weight.data[:]
        stdv = 1. / np.sqrt(layer.weight.size(1))
        layer.weight.data.uniform_(-stdv, stdv)
        if layer.bias is not None:
            layer.bias.data.uniform_(-stdv, stdv)

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float


train_dataset = torchvision.datasets.MNIST('../data', train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.MNIST('../data', train=False, transform=None, target_transform=None, download=True)

# dont use full set

# Standardize data
x_train = np.array(train_dataset.train_data, dtype=np.float)
x_train = x_train.reshape(x_train.shape[0],-1)/255
x_test = np.array(test_dataset.test_data, dtype=np.float)
x_test = x_test.reshape(x_test.shape[0],-1)/255

y_train = np.array(train_dataset.train_labels, dtype=np.int)
y_test  = np.array(test_dataset.test_labels, dtype=np.int)

# dont use full set
#x_train = x_train[:1000, :]
#y_train = y_train[:1000]

T = 300
T_test = 300
input_neurons = 28*28
hidden1_neurons = 500
hidden2_neurons = 300
output_neurons = 10
batch_size = 1024
burnin = 50

layer1 = LIFDenseLayer(in_channels = input_neurons, out_channels = hidden1_neurons, batch_size = batch_size, device = device).to(device)
random_readout1 = FALinear(hidden1_neurons, output_neurons).to(device)
layer2 = LIFDenseLayer(in_channels = hidden1_neurons, out_channels = hidden2_neurons, batch_size = batch_size, device = device).to(device)
random_readout2 = FALinear(hidden2_neurons, output_neurons).to(device)
layer3 = LIFDenseLayer(in_channels = hidden2_neurons, out_channels = output_neurons, batch_size = batch_size, device = device).to(device)

log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
nll_loss = torch.nn.NLLLoss()
opt = torch.optim.Adam([layer1.weights, layer1.bias, layer2.weights, layer2.bias, layer3.weights, layer3.bias], lr=1e-7, betas=[0., .95])

for e in range(300):
    start_time = time.time()
    correct = 0
    total = 0
    for x_local, y_local in sparse_data_generator(x_train, y_train, batch_size, T, shuffle = True, device = device):
        loss_hist = 0
        class_rec = torch.zeros([batch_size, output_neurons]).to(device)
        for t in range(T):
            # run network and random readouts
            out_spikes1 = layer1.forward(x_local[:,t,:])
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)

            if t > burnin:
                rreadout1 = random_readout1(superspike(layer1.U))
                y_log_p1 = log_softmax_fn(rreadout1)
                loss_t = nll_loss(y_log_p1, y_local)

                rreadout2 = random_readout2(superspike(layer2.U))
                y_log_p2 = log_softmax_fn(rreadout2)
                loss_t += nll_loss(y_log_p2, y_local)

                y_log_p3 = log_softmax_fn(superspike(layer3.U))
                loss_t += nll_loss(y_log_p3, y_local)
     
                loss_t.backward()
                opt.step()
                opt.zero_grad()
                loss_hist += loss_t

            class_rec += out_spikes3
        correct += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        total += len(y_local)
    train_time = time.time()


    # compute test accuracy
    tcorrect = 0
    ttotal = 0
    for x_local, y_local in sparse_data_generator(x_test, y_test, batch_size, T_test, shuffle = True, device = device):
        class_rec = torch.zeros([batch_size, output_neurons]).to(device)
        for t in range(T_test):
            out_spikes1 = layer1.forward(x_local[:,t,:])
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            class_rec += out_spikes3
        tcorrect += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        ttotal += len(y_local)

    inf_time = time.time()


    print("Epoch "+str(e)+" | Loss: "+str(np.round(loss_hist.item(),4)) + " Train Acc: "+str(np.round(correct.item()/total, 4)) + " Test Acc: "+str(np.round(tcorrect.item()/ttotal, 4)) + " Train Time: "+str(np.round(train_time-start_time, 4))+"s Inference Time: "+str(np.round(inf_time - train_time, 4)) +"s") 


