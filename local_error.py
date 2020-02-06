import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import pylab as plt
from collections import namedtuple



device = 'cpu'
no_inputs = 100
steps = 400

input_spikes = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.2])).sample([steps,no_inputs]).reshape([steps,no_inputs])
target = np.zeros(T)
target[100]=1; target[200]=1; target[300]=1; 


dense_layer = LIFDenseLayer(in_channels=no_inputs, out_channels=1).to(device)
smooth_step = SmoothStep().apply
mse_loss = torch.nn.MSELoss() # loss needs to be something like that ?
opt = torch.optim.Adam(layer1.parameters(), lr=1e-5, betas=[0., .95])
random_readout = nn.Linear(Nout,1).to(device)


for e in range(100):    
    loss_hist = 0
    for n in range(T):
        state, u = dense_layer.forward(input_spikes[n].unsqueeze(0))
        loss_t = mse_loss(random_readout(smooth_step(u)),target[n]) # for local learning would we use membrane potential instead of spikes ?
        loss_t.backward()
        opt.step()
        opt.zero_grad()
        loss_hist += loss_t
    print(e, loss_hist)







Sin = torch.FloatTensor(snn_utils.spiketrains(N=Nin, T=T, rates = np.ones([Nin])*25)).to(device)
layer1 = LIFDenseLayer(in_channels=Nin, out_channels=1).to(device)
yhat = np.zeros(T)
yhat[100]=1; yhat[200]=1; yhat[300]=1; 
yhat = np.convolve(yhat,np.exp(-np.linspace(0,1,100)/.1))
yhat_t = torch.FloatTensor(yhat).to(device)
plt.plot(yhat_t)




mse_loss = torch.nn.MSELoss()
opt = torch.optim.Adam(layer1.parameters(), lr=1e-4, betas=[0., .95]) #lr is the learning rate 
for e in range(1000):    
    loss_hist = 0
    for n in range(T):
        state, u = layer1.forward(Sin[n].unsqueeze(0))
        loss_t = mse_loss(F.sigmoid(u),yhat_t[n]) # membrane potential with sigmoid non linearity
        loss_t.backward()
        opt.step()
        opt.zero_grad()
        loss_hist += loss_t
    if (e%20)==0: print(e, loss_hist)


# test
Sprobe = np.empty([T,1])
Uprobe = np.empty([T,1])
readProbe = np.empty([T,1])
for n in range(T):
    state, u = layer1.forward(Sin[n])
    Uprobe[n] = u.clone().data.cpu().numpy()
    Sprobe[n] = state.S.clone().data.cpu().numpy()
    readProbe[n] = F.sigmoid(u).data.numpy()
snn_utils.plotLIF(U=readProbe, S=Sprobe);





Nout = 256
layer1 = LIFDenseLayer(in_channels=Nin, out_channels=Nout).to(device)
mse_loss = torch.nn.MSELoss()
opt = torch.optim.Adam(layer1.parameters(), lr=1e-5, betas=[0., .95]) #lr is the learning rate 
random_readout = nn.Linear(Nout,1).to(device)



for e in range(300):    
    loss_hist = 0
    for n in range(T):
        state, u = layer1.forward(Sin[n].unsqueeze(0))
        #This is where we introduce the random readout
        loss_t = mse_loss(random_readout(F.sigmoid(u)),yhat_t[n])
        loss_t.backward()
        opt.step()
        opt.zero_grad()
        loss_hist += loss_t
    if (e%20)==0: print(e, loss_hist)


Sprobe = np.empty([T,Nout])
readProbe = np.empty([T,1])
for n in range(T):
    state, u = layer1.forward(Sin[n])
    Sprobe[n] = state.S.clone().data.cpu().numpy()
    readProbe[n] = random_readout(F.sigmoid(u)).data.cpu().numpy()
    
ax1, ax2 = snn_utils.plotLIF(U=readProbe, S=Sprobe)


plt.plot(readProbe, linewidth=3)
plt.plot(yhat, linewidth=3,alpha=.5)
plt.plot(F.sigmoid(torch.Tensor(Uprobe)).data.numpy())






    
smooth_step = SmoothStep().apply

Nout = 256
layer1 = LIFDenseLayer(in_channels=Nin, out_channels=Nout).to(device)
mse_loss = torch.nn.MSELoss()
opt = torch.optim.Adam(layer1.parameters(), lr=1e-5, betas=[0., .95]) #lr is the learning rate 
random_readout = nn.Linear(Nout,1).to(device)

for e in range(300):    
    loss_hist = 0
    for n in range(T):
        state, u = layer1.forward(Sin[n].unsqueeze(0))
        #This is where we introduce the smooth step
        loss_t = mse_loss(random_readout(smooth_step(u)),yhat_t[n])
        loss_t.backward()
        opt.step()
        opt.zero_grad()
        loss_hist += loss_t
    if (e%20)==0: print(e, loss_hist)


readProbe = np.empty([T,1])

for n in range(T):
    state, u = layer1.forward(Sin[n])
    Sprobe[n] = state.S.clone().data.cpu().numpy()
    readProbe[n] = random_readout(smooth_step(u)).data.cpu().numpy()
    
ax1, ax2 = snn_utils.plotLIF(U=readProbe, S=Sprobe)
plt.plot(readProbe, linewidth=3)
plt.plot(yhat, linewidth=3,alpha=.5)





# layer1 = LIFDenseLayer(in_channels=N, out_channels=1).to(device)
# Uprobe = np.empty([1000,1])
# Pprobe = np.empty([1000,N])
# Qprobe = np.empty([1000,N])
# Rprobe = np.empty([1000,1])
# Sprobe = np.empty([1000,1])
# for n in range(1000):
#     state, u = layer1.forward(Sin[n].to(device))
#     Uprobe[n] = u.clone().data.cpu().numpy()
#     Pprobe[n] = state.P.clone().data.cpu().numpy()
#     Qprobe[n] = state.Q.clone().data.cpu().numpy()
#     Rprobe[n] = state.R.clone().data.cpu().numpy()
#     Sprobe[n] = state.S.clone().data.cpu().numpy()


# snn_utils.plotLIF(U=Rprobe, S=Sprobe, staggering=5);
# plt.plot(Rprobe)

# # learning


# # generating a target






# MNIST experiments
import torch
import torch.nn as nn
import torchvision
import numpy as np

from collections import namedtuple

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



class LIFDenseLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])
    def __init__(self, in_channels, out_channels, batch_size, bias=True, alpha = .9, beta=.85):
        super(LIFDenseLayer, self).__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta

        self.weights = 

        self.P = torch.zeros(batch_size, in_channels)
        self.Q = torch.zeros(batch_size, in_channels)
        self.R = torch.zeros(batch_size, out_channels)
        self.S = torch.zeros(batch_size, out_channels)

        #self.state = self.NeuronState(P=torch.zeros(1,in_channels),
        #                              Q=torch.zeros(1,in_channels),
        #                              R=torch.zeros(1,out_channels),
        #                              S=torch.zeros(1,out_channels))
        self.fc_layer.weight.data.uniform_(-.3, .3)
        self.fc_layer.bias.data.uniform_(-.01, .01)
        
    
        return self 
    
    def forward(self, input_t):
        #remember to put in biases again
        self.P, self.R, self.Q = self.alpha * self.P + self.Q, self.alpha * self.R - self.S, self.beta * self.Q + input_t
        self.U = torch.einsum(",ba->ab", ()) + self.R
        self.S = (self.U>0).float() #threshold

        return self.S


        # state = self.state
        # P = self.alpha*state.P + state.Q
        # R = self.alpha*state.R - state.S
        # Q = self.beta*state.Q + Sin_t
        # # einsum here
        # U = self.fc_layer(P) + R
        # # update the neuronal state
        # S = (U>0).float()
        # #The detach function below avoids the backpropagation of error feedback
        # self.state = self.NeuronState(P=P.detach(), Q=Q.detach(), R=R.detach(), S=S.detach())
        # return self.S #self.state, U



    # lets see if this is valuable
    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = self.NeuronState(P=self.state.P.cuda(device),
                                      Q=self.state.Q.cuda(device),
                                      R=self.state.R.cuda(device),
                                      S=self.state.S.cuda(device))
        return self 
    
    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = self.NeuronState(P=self.state.P.cpu(device),
                                      Q=self.state.Q.cpu(device),
                                      R=self.state.R.cpu(device),
                                      S=self.state.S.cpu(device))


# Here we load the Dataset
train_dataset = torchvision.datasets.MNIST('../data', train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.MNIST('../data', train=False, transform=None, target_transform=None, download=True)

# Standardize data
x_train = np.array(train_dataset.train_data, dtype=np.float)
x_train = x_train.reshape(x_train.shape[0],-1)/255
x_test = np.array(test_dataset.test_data, dtype=np.float)
x_test = x_test.reshape(x_test.shape[0],-1)/255


y_train = np.array(train_dataset.train_labels, dtype=np.int)
y_test  = np.array(test_dataset.test_labels, dtype=np.int)


T = 100


for e in range(10):
    for x_local, y_local in sparse_data_generator(x_train, y_train, 128, T, shuffle = True):
        x_local = x_local.to_dense()
        #for t in range(T):

        break
    break

