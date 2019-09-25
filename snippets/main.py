from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from quantization import quantize

def clip_through(x, min, max):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = torch.clamp(x,min,max) #clamp
    return x + (clipped - x) # x + K.stop_gradient(clipped-x)


def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = torch.round(x)
    rounded_through = x + (rounded - x) # rounded_through = x + K.stop_gradient(rounded - x)
    return rounded_through


def quantize(weights, nb = 16, clip_through=False):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    if clip_through:
        Wq = clip_through(round_through(weights*m),-m,m-1)/m
    else:
        Wq = torch.clamp(round_through(weights*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    return Wq


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, padding = 1)

        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding = 1)

        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding = 1)
        
        self.fc1 = nn.Linear(64*28*28, 500)
        
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        #print(x.shape)
        #import pdb; pdb.set_trace()
        x = F.relu(self.conv1(x))
        #x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        #x = F.max_pool2d(x, 2, 2)
        #import pdb; pdb.set_trace()
        x = x.view(-1, 28*28*64)
        x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def squared_hinge(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    y_true.scatter_(1, labels.view(-1, 1), 1)

    return torch.mean((torch.max(1. - y_true * y_pred, torch.Tensor([0]).to(device)))**2)



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        #import pdb; pdb.set_trace()
        #loss = squared_hinge(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

torch.manual_seed(42)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, num_workers=1)


model = Net().to(device)
from torchsummary import summary
summary(model, (1, 28, 28))


#optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 10):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

        
