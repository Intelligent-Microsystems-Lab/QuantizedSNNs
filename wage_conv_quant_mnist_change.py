from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

import matplotlib.pyplot as plt

torch.set_printoptions(precision=7)

# global quantization variables
global_wb = 2
global_ab = 8
global_gb = 8
global_eb = 8
global_rb = 16

global_beta = 1.5
global_lr = 1

b_size = 64

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
    return 0.5 * torch.sum((y_true.double() - y_pred.double())**2)

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

        if ctx.needs_input_grad[0]: #for sure not nice at all... but welp...
            # propagate quantized error
            grad_input = quant_error.mm(w_quant)
        if ctx.needs_input_grad[1]:
            grad_weight = quant_grad(quant_error.t().mm(input)).float()
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        #import pdb; pdb.set_trace()
        return grad_input, grad_weight, grad_bias, None, None

# Inherit from Function
class clee_conv2d(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, scale, act=False, act_q=False, pool=False, bias=None):
        # prep and save
        w_quant = quant_w(weight, scale)
        input = input.float()
        
        # compute output
        output = F.conv2d(input, w_quant, bias=None, stride=1, padding=0, dilation=1, groups=1)

        relu_mask = torch.ones(output.shape).to(output.device)
        clip_info = torch.ones(output.shape).to(output.device)

        # add pool, relu, quant optionally
        if pool:
            pass # insert pooling here... probably really nasty for everything to come...
        if act:
            output = F.relu(output)
            relu_mask = (output != 0)
        if act_q:
            output, clip_info = quant_act(output)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, w_quant, bias, torch.tensor([pool]), relu_mask, clip_info)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, weight, w_quant, bias, pool, relu_mask, clip_info = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if pool:
            pass # probably something really difficult....
        else:
            quant_error = quant_err(grad_output) * relu_mask.float() * clip_info.float()

        if ctx.needs_input_grad[0]:
            # propagate quantized error
            grad_input = torch.nn.grad.conv2d_input(input.shape, w_quant, quant_error)
        if ctx.needs_input_grad[1]:
            grad_weight = quant_grad(torch.nn.grad.conv2d_weight(input, weight.shape, quant_error)).float()

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()

        self.device = device

        self.padder1 = nn.ZeroPad2d(1)

        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=0, dilation=1, groups=1)
        self.conv1.scale = init_layer_weights(self.conv1.weight, [self.conv1.kernel_size[0], self.conv1.kernel_size[1], self.conv1.in_channels, self.conv1.out_channels]).to(device)

        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=0, dilation=1, groups=1)
        self.conv2.scale = init_layer_weights(self.conv2.weight, [self.conv2.kernel_size[0], self.conv2.kernel_size[1], self.conv2.in_channels, self.conv2.out_channels]).to(device)

        self.fc1 = nn.Linear(36864, 512, bias=False)
        self.fc1.scale = init_layer_weights(self.fc1.weight, [self.fc1.in_features, self.fc1.in_features]).to(device)

        self.fc2 = nn.Linear(512, 10, bias=False)
        self.fc2.scale = init_layer_weights(self.fc2.weight, [self.fc2.in_features, self.fc2.in_features]).to(device)


    def forward(self, x):
        # clip weights
        with torch.no_grad():
            self.conv1.weight.data = clip(self.conv1.weight.data, global_wb)
            self.conv2.weight.data = clip(self.conv2.weight.data, global_wb)
            self.fc1.weight.data = clip(self.fc1.weight.data, global_wb)
            self.fc2.weight.data = clip(self.fc2.weight.data, global_wb)

        # normalizing data between -1 and 1
        x = x.float()
        x = x / 127.5 - 1

        batch_size = x.shape[0]

        # quantize inputs with alpha = 1
        x, _ = quant_act(x)

        # conv1 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, self.conv1.weight, self.conv1.scale, True, True)

        # conv2 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, self.conv2.weight, self.conv2.scale, True, True)
        
        # reshape for fc
        x = x.view(batch_size, -1)

        # fully connected layer #1
        x = clee_LinearFunction.apply(x, self.fc1.weight, self.fc1.scale, True, True)

        # fully connected layer #2
        x = clee_LinearFunction.apply(x, self.fc2.weight, self.fc2.scale, False, False)

        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct_guess = 0
    loss_guess = 0
    count = 0

    for batch_idx, (data, target) in enumerate(train_data(train_loader[0], train_loader[1], b_size)):
        data, target = data.to(device), target.to(device)
        count += len(data)

        optimizer.zero_grad()
        output = model(data)

        loss = SSE(target, output)

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        corr_num = target.argmax(dim=1, keepdim=True)
        correct_guess += pred.eq(corr_num.view_as(pred)).sum().item()
        loss_guess += loss.item()
        print('\rTrain Epoch: {} [{:.0f}%]\tLoss: {:.6f} \tAccuarcy: {:.6f} \r'.format(
        epoch, 100. * batch_idx / 938, loss_guess/(batch_idx+1), 100* correct_guess/count ), end="") # the progress % is fixed for batchsize 64
    return correct_guess/count, loss_guess/(batch_idx+1)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    count_bx = 0 
    with torch.no_grad():
        for data, target in test_data(test_loader[0], test_loader[1], b_size):
            count_bx += 1
            data, target = data.to(device), target.to(device)
            output = model(data)

            #target_cat_hinge = to_cat(target, 10, device)
            #test_loss += SSE(target_cat_hinge.to(device), output)
            test_loss += SSE(target, output)

            pred = output.argmax(dim=1, keepdim=True)
            corr_num = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(corr_num.view_as(pred)).sum().item()

    test_loss /= count_bx

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / 157 )) # the progress % is fixed for batchsize 64
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, 10000, 100. * correct /10000 )) # the progress % is fixed for batchsize 64
    return correct / 10000, test_loss


use_cuda = torch.cuda.is_available()
torch.manual_seed(69)
device = torch.device("cuda" if use_cuda else "cpu")

def loadNPZ(pathNPZ):
  data = np.load(pathNPZ)

  trainX = data['trainX']
  trainY = data['trainY']

  testX = data['testX']
  testY = data['testY']

  label = data['label']
  return trainX, trainY, testX, testY, label

pathNPZ = '../dataSet/MNIST.npz'
numpyTrainX, numpyTrainY, numpyTestX, numpyTestY, label = loadNPZ(pathNPZ)

def train_data(trainX, trainY, batch_size):
    samples_id = np.arange(trainX.shape[0])
    while samples_id.size != 0:
        try:
            rand_index = np.random.choice(np.arange(len(samples_id)), batch_size)
        except:
            rand_sample = np.arange(len(samples_id))
        samples_id = np.delete(samples_id, rand_index)
        yield torch.tensor(trainX[rand_index]).reshape([-1,1,28,28]), torch.tensor(trainY[rand_index])

def test_data(testX, testY, batch_size):
    samples_id = np.arange(testX.shape[0])
    while samples_id.size != 0:
        try:
            next_samples = np.arange(b_size)
        except:
            next_samples = np.arange(len(samples_id))
        samples_id = np.delete(samples_id, next_samples)
        yield torch.tensor(testX[next_samples]).reshape([-1,1,28,28]), torch.tensor(testY[next_samples])


# setting up the model
model = Net(device).to(device)
optimizer = optim.SGD(model.parameters(), lr=1, momentum=0, dampening=0, weight_decay=0, nesterov=False)

# train
teacc, teloss, taacc, taloss = [], [], [], []  
for epoch in range(1, 50):
    acc, lossv = train(model, device, (numpyTrainX, numpyTrainY), optimizer, epoch)
    taacc.append(acc)
    taloss.append(lossv)
    acc, lossv = test(model, device, (numpyTestX, numpyTestY))
    teacc.append(acc)
    teloss.append(lossv)



# graph results
plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.plot(taacc , label="Training Accuracy", color="black")
plt.plot(teacc , label="Test Accuracy", color="blue")
plt.legend(loc = 'best')
plt.title("Accuarcy Pytorch WAGE Acc MNIST")

plt.tight_layout()
plt.savefig("figures/torch_wage_acc_mnist.png")


plt.clf()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(taloss , label="Training Loss", color="black")
plt.plot(teloss , label="Test Loss", color="blue")
plt.legend(loc = 'best')
plt.title("Loss Pytorch WAGE Loss MNIST")

plt.tight_layout()
plt.savefig("figures/torch_wage_loss_mnist.png")

