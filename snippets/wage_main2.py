from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def compute_init_L(fan_in, beta, step_i):
    return torch.max(torch.tensor([torch.sqrt(torch.tensor([6/fan_in])), beta*step_i]))

def step_d(bits): 
    return 2**(1-bits)

def shift(x):
    return 2**torch.round(torch.log2(x))

def clip(x, bits):
    if bits == 1:
        delta = 0.
    else:
        delta = step_d(bits)
    maxv = +1 - delta
    minv = -1 + delta
    return torch.clamp(x, minv, maxv)

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
    rounded_through = x + temp # rounded_through = x + K.stop_gradient(rounded - x)
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
    return 0.5 * torch.sum((y_true - y_pred)**2)


# Inherit from Function
class clee_LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, eb, bias=None):
        ctx.save_for_backward(input, weight, bias, eb)
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

        #
        input, weight, bias, eb = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            import pdb; pdb.set_trace()
            eq = qc(grad_output/shift(torch.max(torch.abs(grad_output))), int(eb))

            # compute grad L wrt. w
            grad_weight = eq.t().mm(input)

            # quantize updates
            grad_weight = 1 * grad_weight/shift(torch.max(torch.abs(grad_weight)))
            #grad_w_floor = torch.floor(grad_w)
            #prob_b = grad_w - grad_w_floor
            #grad_weight = step_d(gb) * torch.sign(grad_w) * (grad_w_floor + torch.bernoulli(prob_b))            

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        #import pdb; pdb.set_trace()
        return grad_input, grad_weight, grad_bias

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # quantization variables
        self.wb = 8
        self.ab = 8
        self.gb = 8
        self.eb = 8

        self.beta = 2
        self.lr = 1


        self.fc1 = nn.Linear(28*28, 10, bias=False)

        # initialize quantized weights
        L = compute_init_L(fan_in = 28*28, beta = self.beta, step_i = step_d(self.gb))
        torch.nn.init.uniform_(self.fc1.weight, a = -L, b = L)
        self.fc1.weight = torch.nn.Parameter(qc(self.fc1.weight, self.gb).to(device), requires_grad = True)



    def forward(self, x):
        # clip weights -> last step from the grad update
        self.fc1.weight = torch.nn.Parameter(clip(self.fc1.weight, self.gb))

        # quantize inputs with alpha = 1
        x = qc(x, self.ab)
        
        # # fully connected layer
        x = x.view(-1, 28*28)
        x = clee_LinearFunction.apply(x, self.fc1.weight, torch.tensor([self.eb]))

        # quantize activations
        L = compute_init_L(fan_in = 28*28, beta = self.beta, step_i = step_d(self.ab))
        alpha = torch.max(torch.tensor([shift(L/(self.beta*step_d(self.ab))), 1]))
        x = qc(x/alpha, self.ab)

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
        loss = SSE(target_cat_hinge.to(device), output)

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct_guess += pred.eq(target.view_as(pred)).sum().item()
        loss_guess += loss.item()
        print('\r Train Epoch: {} [{:.0f}%]\tLoss: {:.6f} \tAccuarcy: {:.6f} \r'.format(
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

# load data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=64, shuffle=False, num_workers = 2)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=64, shuffle=False, num_workers = 2)


# setting up the model
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=1, momentum=0, dampening=0, weight_decay=0, nesterov=False)


# print model summary
#from torchsummary import summary
#summary(model, (1, 28, 28))


# train
teacc, teloss, taacc, taloss = [], [], [], []  
for epoch in range(1, 10):
    acc, lossv = train(model, device, train_loader, optimizer, epoch)
    taacc.append(acc)
    taloss.append(lossv)
    acc, lossv = test(model, device, test_loader)
    teacc.append(acc)
    teloss.append(lossv)



# graph results
# plt.clf()
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.plot(taacc , label="Training Accuracy", color="black")
# plt.plot(teacc , label="Test Accuracy", color="blue")
# plt.legend(loc = 'best')
# plt.title("Accuarcy Pytorch No Quantized Learning")

# plt.tight_layout()
# plt.savefig("figures/torch_acc_nql.png")


# plt.clf()
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.plot(taloss , label="Training Loss", color="black")
# plt.plot(teloss , label="Test Loss", color="blue")
# plt.legend(loc = 'best')
# plt.title("Loss Pytorch No Quantized Learning")

# plt.tight_layout()
# plt.savefig("figures/torch_loss_nql.png")
