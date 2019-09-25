from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# global quantization variables
global_wb = 2
global_ab = 8
global_gb = 8
global_eb = 8

global_beta = 1.5
global_lr = 1


def angel_between(vec1, vec2):
    x = torch.einsum("i,i", vec1, vec2)/(vec1.norm(p=2)*vec2.norm(p=2))
    if x < 0:
        print("{:.9f}".format(float(x)+1))
    else:
        print("{:.9f}".format(float(x)-1))
    return torch.acos(torch.clamp(x, min=-1, max=1))


def compute_init_L(fan_in, beta, step_i):
    return torch.max(torch.tensor([torch.sqrt(torch.tensor([6/fan_in])), beta*step_i]))

def step_d(bits): 
    return 2**(1-bits)

def shift(x):
    if x == 0:
        return 1
    return 2**round_through(torch.log2(x))

def clip(x, bits):
    if bits == 1:
        delta = 0.
    else:
        delta = step_d(bits)
    maxv = +1 - delta
    minv = -1 + delta
    return torch.clamp(x, float(minv), float(maxv))

def quant(x, bits):
    if bits == 1: # BNN
        return torch.sign(x)
    else:
        return round_through(x/step_d(bits)) * step_d(bits)

def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = torch.round(x)
    with torch.no_grad():
        temp = (rounded - x)
    rounded_through = x + temp
    return rounded_through

#combining clip and quant for convenience
def qc(x, bits):
    return clip(quant(x, bits) , bits)

def to_cat(inp_tensor, num_class, device):
    out_tensor = torch.zeros([inp_tensor.shape[0], num_class], device=device)
    out_tensor[torch.arange(inp_tensor.shape[0]).to(device), torch.tensor(inp_tensor, dtype = int, device=device)] = 1
    return out_tensor

# sum of square errors
def SSE(y_true, y_pred):
    #import pdb; pdb.set_trace()
    return 0.5 * torch.sum((y_true - y_pred)**2)

def SSM(y_true, y_pred):
    #import pdb; pdb.set_trace()
    return 0.5 * torch.mean((y_true - y_pred)**2)

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
        alpha = shift(torch.max(torch.abs(grad_output)))

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            #grad_input = grad_output.mm(weight)
            grad_input = qc(grad_output/alpha, global_eb).mm(weight)
        if ctx.needs_input_grad[1]:
            shift_gw = global_lr * qc(grad_output/alpha, global_eb).t().mm(input) 

            alpha = shift(torch.max(torch.abs(shift_gw)))
            grad_weight = shift_gw/alpha

            grad_w_floor = torch.floor(grad_weight/step_d(global_gb))
            prob_b = grad_weight/step_d(global_gb) - grad_w_floor
            grad_weight = step_d(global_gb) * torch.sign(grad_weight) * (torch.abs(grad_w_floor) + torch.bernoulli(prob_b))
            
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
        alpha = shift(torch.max(torch.abs(grad_output)))
        quant_error = qc(grad_output/alpha, global_eb)

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            #grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output)
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, quant_error)
        if ctx.needs_input_grad[1]:
            shift_gw = global_lr * torch.nn.grad.conv2d_weight(input, weight.shape, quant_error)

            alpha = shift(torch.max(torch.abs(shift_gw)))
            grad_weight = shift_gw/alpha

            grad_w_floor = torch.floor(grad_weight/step_d(global_gb))
            prob_b = grad_weight/step_d(global_gb) - grad_w_floor
            grad_weight = step_d(global_gb) * torch.sign(grad_weight) * (torch.abs(grad_w_floor) + torch.bernoulli(prob_b))
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


def init_layer_weights(weights_layer, fan_in):
    L = compute_init_L(fan_in = inp_dim, beta = global_beta, step_i = step_d(global_wb))
    torch.nn.init.uniform_(weights_layer, a = -L, b = L)
    weights_layer = torch.nn.Parameter(qc(weights_layer, global_gb).to(device), requires_grad = True)

def quant_act(x, fan_in):
    L = compute_init_L(fan_in = fan_in, beta = global_beta, step_i = step_d(global_ab))
    alpha = torch.max(torch.tensor([shift(L/(global_beta*step_d(global_ab))), 1]))
    return qc(x/alpha, global_ab)

class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        w_step = step_d(global_wb)

        self.device = device

        self.conv1 = nn.Conv2d(1, 64, 5, stride=1, padding=0, dilation=1, groups=1)
        self.conv1.fan_in = self.conv1.in_channels * self.conv1.kernel_size[0] * self.conv1.kernel_size[1]
        init_layer_weights(self.conv1.weight, self.conv1.fan_in)

        self.conv2 = nn.Conv2d(64, 64, 5, stride=1, padding=0, dilation=1, groups=1)
        self.conv2.fan_in = self.conv2.in_channels * self.conv2.kernel_size[0] * self.conv2.kernel_size[1]
        init_layer_weights(self.conv2.weight, self.conv2.fan_in)

        self.conv3 = nn.Conv2d(64, 64, 5, stride=1, padding=0, dilation=1, groups=1)
        self.conv3.fan_in = self.conv3.in_channels * self.conv3.kernel_size[0] * self.conv3.kernel_size[1]
        init_layer_weights(self.conv3.weight, self.conv3.fan_in)

        self.fc1 = nn.Linear(784, 100, bias=False)
        self.fc1.fan_in = self.fc1.in_features
        init_layer_weights(self.fc1.weight, self.fc1.fan_in)

        self.fc2 = nn.Linear(100, 50, bias=False)
        self.fc2.fan_in = self.fc2.in_features
        init_layer_weights(self.fc2.weight, self.fc2.fan_in)

        self.fc3 = nn.Linear(50, 10, bias=False)
        self.fc3.fan_in = self.fc3.in_features
        init_layer_weights(self.fc3.weight, self.fc2.fan_in)


    def forward(self, x):
        # Average pooling is avoided because mean operations will increase precision demand.
        batch_size = x.shape[0]

        # clip weights without destroying autograd -> last step from the grad update
        delta = step_d(global_gb)
        self.conv1.weight.data.clamp_(min = -1 + float(delta), max = +1 - float(delta))
        self.conv2.weight.data.clamp_(min = -1 + float(delta), max = +1 - float(delta))
        self.conv3.weight.data.clamp_(min = -1 + float(delta), max = +1 - float(delta))
        self.fc1.weight.data.clamp_(min = -1 + float(delta), max = +1 - float(delta))
        self.fc2.weight.data.clamp_(min = -1 + float(delta), max = +1 - float(delta))
        #self.fc3.weight.data.clamp_(min = -1 + float(delta), max = +1 - float(delta))

        # quantize inputs with alpha = 1
        x = qc(x, global_ab)

        # conv1 layer
        #x = clee_conv2d.apply(x, qc(self.conv1.weight, global_wb))
        #x = quant_act(x, self.conv1.fan_in)
        #x = F.relu(x) # ReLu

        # conv2 layer
        #x = clee_conv2d.apply(x, qc(self.conv2.weight, global_wb))
        #x = quant_act(x, inp_dim)
        #x = F.relu(x) # ReLu

        # conv3 layer
        #x = clee_conv2d.apply(x, qc(self.conv3.weight, global_wb))
        #x = quant_act(x, inp_dim)
        #x = F.relu(x) # ReLu

        
        # reshape for fc
        x = x.view(batch_size, -1)

        # fully connected layer #1
        x = clee_LinearFunction.apply(x, qc(self.fc1.weight, global_wb))
        x = quant_act(x, self.fc1.fan_in)
        x = F.relu(x) # Relu

        # fully connected layer #2
        x = clee_LinearFunction.apply(x, qc(self.fc2.weight, global_wb))
        x = quant_act(x, self.fc2.fan_in)
        x = F.relu(x) # Relu

        # fully connected layer #3
        x = clee_LinearFunction.apply(x, qc(self.fc3.weight, global_wb))
        x = quant_act(x, self.fc3.fan_in)
        x = F.relu(x) # Relu
        
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

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct_guess += pred.eq(target.view_as(pred)).sum().item()
        loss_guess += loss.item()
        print('\rTrain Epoch: {} [{:.0f}%]\tLoss: {:.6f} \tAccuarcy: {:.6f} \r'.format(
        epoch, 100. * batch_idx / len(train_loader), loss_guess/(batch_idx+1), 100* correct_guess/count ), end="")
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

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= count_bx

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset), test_loss


use_cuda = torch.cuda.is_available()
torch.manual_seed(69)
device = torch.device("cuda" if use_cuda else "cpu")


inp_dim = 28*28
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=256, shuffle=True, num_workers = 2)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=256, shuffle=True, num_workers = 2)

# inp_dim = 3072
# train_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor()
#                    ])),
#     batch_size=256, shuffle=True, num_workers = 2)
# test_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor()
#                    ])),
#     batch_size=256, shuffle=True, num_workers = 2)


# setting up the model
model = Net(device).to(device)
optimizer = optim.SGD(model.parameters(), lr=1, momentum=0, dampening=0, weight_decay=0, nesterov=False)


# print model summary
#from torchsummary import summary
#summary(model, (1, 28, 28)) 


# train
teacc, teloss, taacc, taloss = [], [], [], []  
for epoch in range(1, 50):
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
plt.title("Accuarcy Pytorch WAGE Acc")

plt.tight_layout()
plt.savefig("figures/torch_wage_acc.png")


plt.clf()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(taloss , label="Training Loss", color="black")
plt.plot(teloss , label="Test Loss", color="blue")
plt.legend(loc = 'best')
plt.title("Loss Pytorch WAGE Loss")

plt.tight_layout()
plt.savefig("figures/torch_wage_loss.png")
