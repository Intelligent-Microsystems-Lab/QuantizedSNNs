from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np

import pickle

import matplotlib.pyplot as plt


# global quantization variables
global_wb = 2
global_ab = 8
global_gb = 8
global_eb = 8
global_rb = 16

global_beta = 1.5
global_lr = 8


def angle_between(vec1, vec2):
    x = torch.einsum("i,i", vec1, vec2)/(vec1.norm(p=2)*vec2.norm(p=2))
    if x < 0:
        print("{:.9f}".format(float(x)+1))
    else:
        print("{:.9f}".format(float(x)-1))
    return torch.acos(torch.clamp(x, min=-1, max=1))

def to_cat(inp_tensor, num_class, device):
    out_tensor = torch.zeros([inp_tensor.shape[0], num_class], device=device)
    out_tensor[torch.arange(inp_tensor.shape[0]).to(device), torch.tensor(inp_tensor, dtype = int, device=device)] = 1
    return out_tensor


# Quant Functions

def step_d(bits): 
    return 2.0 ** (bits - 1)

def shift(x):
    if x == 0:
        print("this should not happen")
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
        diff = (y - x) #maybe do scale here

    if scale <= 1.8:
        return x + diff
    return (x + diff)/scale

def quant_act(x):
    x = clip(x, global_ab)
    with torch.no_grad():
        y = quant(x, global_ab)
        diff = y - x
    return x + diff

def quant_grad(x):
    xmax = torch.max(torch.abs(x))
    x = x / shift(xmax)

    norm = quant(global_lr * x, global_rb)

    norm_sign = torch.sign(norm)
    norm_abs = torch.abs(norm)
    norm_int = torch.floor(norm_abs)
    norm_float = norm_abs - norm_int
    rand_float = torch.FloatTensor(x.shape).uniform_(0,1).to(x.device)
    norm = norm_sign * ( norm_int + 0.5 * (torch.sign(norm_float - rand_float) + 1) )

    return norm / step_d(global_gb)

def quant_err(x):
    alpha = shift(torch.max(torch.abs(x)))
    return quant(clip(x / alpha, global_eb), global_eb)

def init_layer_weights(weights_layer, shape, factor=2):
    fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])

    limit = torch.sqrt(torch.tensor([3*factor/fan_in]))
    Wm = global_beta/step_d(torch.tensor([float(global_wb)]))
    scale = 2 ** round(math.log(Wm / limit, 2.0))
    scale = scale if scale > 1 else 1.0
    limit = Wm if Wm > limit else limit

    torch.nn.init.uniform_(weights_layer, a = -float(limit), b = float(limit))
    return torch.tensor([float(scale)])

# sum of square errors
def SSE(y_true, y_pred):
    return 0.5 * torch.sum((y_true - y_pred)**2)

# Inherit from Function
class clee_LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        quant_error = quant_err(grad_output)

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            # propagate quantized error
            grad_input = quant_error.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = quant_grad(quant_error.t().mm(input))
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

# Inherit from Function
class clee_conv2d(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        quant_error = quant_err(grad_output)

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            # propagate quantized error
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, quant_error)
        if ctx.needs_input_grad[1]:
            grad_weight = quant_grad(torch.nn.grad.conv2d_weight(input, weight.shape, quant_error))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias





class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        w_step = step_d(global_wb)

        self.device = device

        self.mpool1 = nn.MaxPool2d(2, stride=2)
        self.padder1 = nn.ZeroPad2d(1)
        self.padder2 = nn.ZeroPad2d(2)

        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv1.scale = init_layer_weights(self.conv1.weight, [self.conv1.kernel_size[0], self.conv1.kernel_size[1], self.conv1.in_channels, self.conv1.out_channels]).to(device)

        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv2.scale = init_layer_weights(self.conv2.weight, [self.conv2.kernel_size[0], self.conv2.kernel_size[1], self.conv2.in_channels, self.conv2.out_channels]).to(device)


        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv3.scale = init_layer_weights(self.conv3.weight, [self.conv3.kernel_size[0], self.conv3.kernel_size[1], self.conv3.in_channels, self.conv3.out_channels]).to(device)


        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv4.scale = init_layer_weights(self.conv4.weight, [self.conv4.kernel_size[0], self.conv4.kernel_size[1], self.conv4.in_channels, self.conv4.out_channels]).to(device)


        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv5.scale = init_layer_weights(self.conv5.weight, [self.conv5.kernel_size[0], self.conv5.kernel_size[1], self.conv5.in_channels, self.conv5.out_channels]).to(device)


        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv6.scale = init_layer_weights(self.conv6.weight, [self.conv6.kernel_size[0], self.conv6.kernel_size[1], self.conv6.in_channels, self.conv6.out_channels]).to(device)


        # 784 36864 ... 16384
        self.fc1 = nn.Linear(8192, 1024, bias=False)
        self.fc1.scale = init_layer_weights(self.fc1.weight, [self.fc1.in_features, self.fc1.in_features]).to(device)

        self.fc2 = nn.Linear(1024, 10, bias=False)
        self.fc2.scale = init_layer_weights(self.fc2.weight, [self.fc2.in_features, self.fc2.in_features]).to(device)


    def forward(self, x):
        batch_size = x.shape[0]
        #import pdb; pdb.set_trace()

        #x = x*2-1
        # quantize inputs with alpha = 1
        #import pdb; pdb.set_trace()
        x = quant_act(x)

        # conv1 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, quant_w(self.conv1.weight, self.conv1.scale))
        x = self.padder2(x)
        x = self.mpool1(x) # pooling
        x = F.relu(x) # ReLu
        x = quant_act(x) 

        # conv2 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, quant_w(self.conv2.weight, self.conv2.scale))
        x = self.padder2(x)
        x = self.mpool1(x) # pooling
        x = F.relu(x) # ReLu
        x = quant_act(x)

        # conv3 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, quant_w(self.conv3.weight, self.conv3.scale))
        x = self.padder2(x)
        x = self.mpool1(x) # pooling
        x = F.relu(x) # ReLu
        x = quant_act(x) 

        # conv4 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, quant_w(self.conv4.weight, self.conv4.scale))
        x = self.padder2(x)
        x = self.mpool1(x) # pooling
        x = F.relu(x) # ReLu
        x = quant_act(x)

        # conv5 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, quant_w(self.conv5.weight, self.conv5.scale))
        x = self.padder2(x)
        x = self.mpool1(x) # pooling
        x = F.relu(x) # ReLu
        x = quant_act(x) 

        # conv6 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, quant_w(self.conv6.weight, self.conv6.scale))
        x = self.padder2(x)
        x = self.mpool1(x) # pooling
        x = F.relu(x) # ReLu
        x = quant_act(x)
        
        # reshape for fc
        x = x.view(batch_size, -1)

        # fully connected layer #1
        x = clee_LinearFunction.apply(x, quant_w(self.fc1.weight, self.fc1.scale))
        x = F.relu(x) # Relu
        x = quant_act(x)

        # fully connected layer #2
        x = clee_LinearFunction.apply(x, quant_w(self.fc2.weight, self.fc2.scale))
        #x = F.relu(x) # Relu
        #x = quant_act(x)

        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct_guess = 0
    loss_guess = 0
    count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        count += len(data)

        optimizer.zero_grad()
        output = model(data)

        target_cat_hinge = to_cat(target, 10, device)
        loss = SSE(target_cat_hinge.to(device), output) # for now, SSE? 
        #loss = SSE(target.float().to(device), output)

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        #target = target.argmax(dim=1, keepdim=True)
        correct_guess += pred.eq(target.view_as(pred)).sum().item()
        loss_guess += loss.item()
        print('\rTrain Epoch: {} [{:.0f}%]\tLoss: {:.6f} \tAccuarcy: {:.6f} \r'.format(epoch, 100. * batch_idx / len(train_loader), loss_guess/(batch_idx+1), 100* correct_guess/count ), end="")
    return correct_guess/count, loss_guess/(batch_idx+1)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    count_bx = 0 
    with torch.no_grad():
        for data, target in test_loader:
            count_bx += 1
            data, target = data.to(device), target.to(device)
            output = model(data)

            target_cat_hinge = to_cat(target, 10, device)
            test_loss += SSE(target_cat_hinge.to(device), output)
            #test_loss += SSE(target.float().to(device), output)

            #target = target.argmax(dim=1, keepdim=True)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= count_bx

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset), test_loss


use_cuda = torch.cuda.is_available()
torch.manual_seed(69)
device = torch.device("cuda" if use_cuda else "cpu")


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform=transforms.Compose([
                       #transforms.RandomCrop([40, 40], pad_if_needed=True),
                       transforms.RandomHorizontalFlip(),
                       #transforms.RandomCrop([32,32]),
                       transforms.ToTensor(),
                       #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=128, shuffle=False, num_workers = 2)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])),
    batch_size=100, shuffle=False, num_workers = 2)


# setting up the model
model = Net(device).to(device)
optimizer = optim.SGD(model.parameters(), lr=1, momentum=0, dampening=0, weight_decay=0, nesterov=False)

# train
teacc, teloss, taacc, taloss = [], [], [], []  
for epoch in range(300):
    # learning rate scheduler
    if (epoch == 2) or (epoch == 250):
        global_lr /= 8
    acc, lossv = train(model, device, train_loader, optimizer, epoch)
    taacc.append(acc)
    taloss.append(lossv)
    acc, lossv = test(model, device, test_loader)
    teacc.append(acc)
    teloss.append(lossv)



# graph results
plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.plot(taacc , label="Training Accuracy", color="black")
plt.plot(teacc , label="Test Accuracy", color="blue")
plt.legend(loc = 'best')
plt.title("Accuarcy Pytorch WAGE Acc CIFAR10")

plt.tight_layout()
plt.savefig("figures/torch_wage_acc_cifar10.png")


plt.clf()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(taloss , label="Training Loss", color="black")
plt.plot(teloss , label="Test Loss", color="blue")
plt.legend(loc = 'best')
plt.title("Loss Pytorch WAGE Loss CIFAR10")

plt.tight_layout()
plt.savefig("figures/torch_wage_loss_cifar10.png")
