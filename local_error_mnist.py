import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
import time
import numpy as np

import matplotlib.pyplot as plt

def clee_spikes(T, rates):
    spikes = np.ones((T, + np.prod(rates.shape)))        
    spikes[np.random.binomial(1, (1000. - rates.flatten())/1000, size=(T, np.prod(rates.shape))).astype('bool')] = 0
    return spikes.T.reshape((rates.shape + (T,)))

def sparse_data_generator(X, y, batch_size, nb_steps, samples, max_hertz, shuffle=True, device=torch.device("cpu")):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors. 

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """
    sample_idx = torch.randperm(len(X))[:samples]
    number_of_batches = int(np.ceil(samples/batch_size))
    nb_steps = int(nb_steps)

    counter = 0
    while counter<number_of_batches:
        if counter == number_of_batches:
            cur_sample = sample_idx[batch_size*counter:]
        else:
            cur_sample = sample_idx[batch_size*counter:batch_size*(counter+1)]

        X_batch = np.zeros((cur_sample.shape[0],) + X.shape[1:] + (nb_steps,))
        for i,idx in enumerate(cur_sample):
            X_batch[i] = clee_spikes(T = nb_steps, rates=max_hertz*X[idx,:]).astype(np.float32)

        X_batch = torch.from_numpy(X_batch).float()
        y_batch = y[cur_sample]
        try:
            yield X_batch.to(device), y_batch.to(device)
            counter += 1
        except StopIteration:
            return


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

class LinearFunctional(torch.autograd.Function):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    @staticmethod
    def forward(context, input, weight, bias=None):
        context.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, bias = context.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class LinearLayer(nn.Module):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    def __init__(self, input_features, output_features, bias=True):
        super(LinearLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # weight and bias for forward pass
        self.weights = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        stdv = 1. / np.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return LinearFunctional.apply(input, self.weights, self.bias)



class LIFDenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, tau_syn, tau_mem, tau_ref, delta_t, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float):
        super(LIFDenseLayer, self).__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.thr = thr

        if tau_syn.shape[0] == 2:
            self.beta = torch.exp( -delta_t / torch.Tensor(self.in_channels).uniform_(tau_syn[0], tau_syn[0]).to(device))
        else:
            self.beta = torch.Tensor([torch.exp( - delta_t / tau_syn)]).to(device)
        if tau_mem.shape[0] == 2:
            self.alpha = torch.exp( -delta_t / torch.Tensor(self.in_channels).uniform_(tau_mem[0], tau_mem[0]).to(device))
        else:
            self.alpha = torch.Tensor([torch.exp( - delta_t / tau_mem)]).to(device)

        if tau_ref.shape[0] == 2:
            self.gamma = torch.exp( -delta_t / torch.Tensor(self.out_channels).uniform_(tau_ref[0], tau_ref[0]).to(device))
        else:
            self.gamma = torch.Tensor([torch.exp( - delta_t / tau_ref)]).to(device)


        self.weights = nn.Parameter(torch.empty((self.in_channels, self.out_channels),  device=device, dtype=dtype, requires_grad=True))
        torch.nn.init.uniform_(self.weights, a = -.3, b = .3)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels,  device=device, dtype=dtype, requires_grad=True))
            torch.nn.init.uniform_(self.bias, a = -.01, b = .01)
        else:
            self.register_parameter('bias', None)

    def state_init(self, batch_size):
        self.P = torch.zeros(batch_size, self.in_channels).detach().to(device)
        self.Q = torch.zeros(batch_size, self.in_channels).detach().to(device)
        self.R = torch.zeros(batch_size, self.out_channels).detach().to(device)
        self.S = torch.zeros(batch_size, self.out_channels).detach().to(device)
        self.U = torch.zeros(batch_size, self.out_channels).detach().to(device)

    
    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t

        self.U = torch.einsum("ab,bc->ac", (self.P, self.weights)) + self.bias + self.R
        self.S = (self.U>self.thr).float()

        return self.S


class LIFConvLayer(nn.Module):
    def __init__(self, inp_shape, kernel_size, out_channels, tau_syn, tau_mem, tau_ref, delta_t, pooling = 1, padding = 0, bias=True, thr = 1, device=torch.device("cpu"), dtype = torch.float):
        super(LIFConvLayer, self).__init__()   
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.out_channels = out_channels  

        self.padding = padding
        self.pooling = pooling

        self.mpoolF = nn.MaxPool2d(kernel_size = self.pooling, stride = self.pooling, padding = (self.pooling-1)//2, return_indices=False) 
        
        self.weights = nn.Parameter(torch.empty((self.out_channels, inp_shape[0],  self.kernel_size, self.kernel_size),  device=device, dtype=dtype, requires_grad=True))
        torch.nn.init.uniform_(self.weights, a = -.3, b = .3)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype, requires_grad=True))
            torch.nn.init.uniform_(self.bias, a = -.01, b = .01)
        else:
            self.register_parameter('bias', None)

        self.out_shape = self.mpoolF(F.conv2d(input = torch.zeros((1,)+self.inp_shape).to(device), weight = self.weights, bias=self.bias, padding = self.padding)).shape[1:]
        self.thr = thr

        if tau_syn.shape[0] == 2:
            self.beta = torch.exp( -delta_t / torch.Tensor(self.inp_shape).uniform_(tau_syn[0], tau_syn[0]).to(device))
        else:
            self.beta = torch.Tensor([torch.exp( - delta_t / tau_syn)]).to(device)
        if tau_mem.shape[0] == 2:
            self.alpha = torch.exp( -delta_t / torch.Tensor(self.inp_shape).uniform_(tau_mem[0], tau_mem[0]).to(device))
        else:
            self.alpha = torch.Tensor([torch.exp( - delta_t / tau_mem)]).to(device)

        if tau_ref.shape[0] == 2:
            self.gamma = torch.exp( -delta_t / torch.Tensor(self.out_shape).uniform_(tau_ref[0], tau_ref[0]).to(device))
        else:
            self.gamma = torch.Tensor([torch.exp( - delta_t / tau_ref)]).to(device)


        

    def state_init(self, batch_size):
        self.P = torch.zeros((batch_size,) + self.inp_shape).detach().to(device)
        self.Q = torch.zeros((batch_size,) + self.inp_shape).detach().to(device)
        self.R = torch.zeros((batch_size,) + self.out_shape).detach().to(device)
        self.S = torch.zeros((batch_size,) + self.out_shape).detach().to(device)
        self.U = torch.zeros((batch_size,) + self.out_shape).detach().to(device)

    
    def forward(self, input_t):
        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.gamma * self.R - self.S, self.beta * self.Q + input_t
        self.U = self.mpoolF(F.conv2d(input = self.P, weight = self.weights, bias=self.bias, padding = self.padding)) + self.R
        self.S = (self.U>self.thr).float()

        return self.S




# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float


train_dataset = torchvision.datasets.MNIST('../data', train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.MNIST('../data', train=False, transform=None, target_transform=None, download=True)

# Standardize data
x_train = train_dataset.data.type(dtype)/255
x_train = x_train.reshape((x_train.shape[0],) + (1,) + x_train.shape[1:])
x_test = test_dataset.data.type(dtype)/255
x_test = x_test.reshape((x_test.shape[0],) + (1,) + x_test.shape[1:])

y_train = train_dataset.targets
y_test  = test_dataset.targets


# fixed subsampling
# train: 300 samples per class -> 3000
# test: 103 samples per class -> 1030 (a wee more than 1024)
index_list_train = []
index_list_test = []
for i in range(10):
    index_list_train.append((y_train == i).nonzero()[:300])
    index_list_test.append((y_test == i).nonzero()[:103])
index_list_train = torch.cat(index_list_train).reshape([3000])
index_list_test = torch.cat(index_list_test).reshape([1030])

x_train = x_train[index_list_train, :]
x_test = x_test[index_list_test, :]
y_train = y_train[index_list_train]
y_test = y_test[index_list_test]



ms = 1e-3
delta_t = 1*ms

T = 500*ms
T_test = 1000*ms
burnin = 50*ms
batch_size = 64
output_neurons = 10

tau_mem = torch.Tensor([5*ms, 35*ms]).to(device)
tau_syn = torch.Tensor([5*ms, 10*ms]).to(device)
tau_ref = torch.Tensor([2.86*ms]).to(device)
thr = torch.Tensor([1.]).to(device)


# dense architecture
# input_neurons = 28*28
# hidden1_neurons = 500
# hidden2_neurons = 300
# output_neurons = 10

# layer1 = LIFDenseLayer(in_channels = input_neurons, out_channels = hidden1_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)
# random_readout1 = LinearLayer(hidden1_neurons, output_neurons).to(device)

# layer2 = LIFDenseLayer(in_channels = hidden1_neurons, out_channels = hidden2_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)
# random_readout2 = LinearLayer(hidden2_neurons, output_neurons).to(device)

# layer3 = LIFDenseLayer(in_channels = hidden2_neurons, out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)

dropout_learning = nn.Dropout(p=0.)

layer1 = LIFConvLayer(inp_shape = x_train.shape[1:], kernel_size = 7, out_channels = 16, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout1 = LinearLayer(np.prod(layer1.out_shape), output_neurons).to(device)

layer2 = LIFConvLayer(inp_shape = layer1.out_shape, kernel_size = 7, out_channels = 24, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 1, padding = 2, thr = thr, device = device).to(device)
random_readout2 = LinearLayer(np.prod(layer2.out_shape), output_neurons).to(device)

layer3 = LIFConvLayer(inp_shape = layer2.out_shape, kernel_size = 7, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout3 = LinearLayer(np.prod(layer3.out_shape), output_neurons).to(device)

layer4 = LIFDenseLayer(in_channels = np.prod(layer3.out_shape), out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)

log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
nll_loss = torch.nn.NLLLoss()
params = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters()) + list(layer4.parameters()) 
opt = torch.optim.Adam(params, lr=1e-9, betas=[0., .95])

for e in range(300):
    start_time = time.time()
    correct = 0
    total = 0
    for x_local, y_local in sparse_data_generator(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, samples = 3000, max_hertz = 50, shuffle = True, device = device):
        class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)

        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])
        layer4.state_init(x_local.shape[0])
        loss_hist = 0

        # burnin
        for t in range(int(burnin/ms)):
            out_spikes1 = layer1.forward(x_local[:,:,:,:,t])
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)


        # training
        for t in range(int(burnin/ms), int(T/ms)):
            out_spikes1 = layer1.forward(x_local[:,:,:,:,t])
            rreadout1 = random_readout1(dropout_learning(smoothstep(layer1.U.reshape([x_local.shape[0], np.prod(layer1.out_shape)]))))
            y_log_p1 = log_softmax_fn(rreadout1)
            loss_t = nll_loss(y_log_p1, y_local)

            out_spikes2 = layer2.forward(out_spikes1)
            rreadout2 = random_readout2(dropout_learning(smoothstep(layer2.U.reshape([x_local.shape[0], np.prod(layer2.out_shape)]))))
            y_log_p2 = log_softmax_fn(rreadout2)
            loss_t += nll_loss(y_log_p2, y_local)

            out_spikes3 = layer3.forward(out_spikes2)
            rreadout3 = random_readout3(dropout_learning(smoothstep(layer3.U.reshape([x_local.shape[0], np.prod(layer3.out_shape)]))))
            y_log_p3 = log_softmax_fn(rreadout3)
            loss_t += nll_loss(y_log_p3, y_local)

            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)
            y_log_p4 = log_softmax_fn(smoothstep(layer4.U))
            loss_t += nll_loss(y_log_p4, y_local)
 
            # introduce regularizer

            loss_t.backward()
            opt.step()
            opt.zero_grad()
            loss_hist += loss_t

            class_rec += out_spikes4

        correct += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        total += len(y_local)
    train_time = time.time()


    # compute test accuracy
    tcorrect = 0
    ttotal = 0
    for x_local, y_local in sparse_data_generator(x_test, y_test, batch_size = batch_size, nb_steps = T_test/ms, samples = 1030, max_hertz = 50, shuffle = True, device = device):
        class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)
        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])
        layer4.state_init(x_local.shape[0])

        for t in range(int(T_test/ms)):
            out_spikes1 = layer1.forward(x_local[:,:,:,:,t])
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)
            class_rec += out_spikes4
        tcorrect += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        ttotal += len(y_local)
    inf_time = time.time()

    print("Epoch {0} | Loss: {1:.4f} Train Acc: {2:.4f} Test Acc: {3:.4f} Train Time: {4:.4f}s Inference Time: {5:.4f}s".format(e+1, loss_hist.item(), correct.item()/total, tcorrect.item()/ttotal, train_time-start_time, inf_time - train_time)) 



