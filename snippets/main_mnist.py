from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# from https://github.com/keras-team/keras/blob/master/keras/losses.py
def squared_hinge(y_true, y_pred):
    #import pdb; pdb.set_trace()
    return torch.mean((torch.max(1. - y_true * y_pred, torch.Tensor([0.]).to("cuda")))**2)


def clip_through(x, min_val, max_val):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = torch.clamp(x, min_val, max_val) #clamp
    with torch.no_grad():
        temp = (clipped - x)
    return x + temp # x + K.stop_gradient(clipped-x)


def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = torch.round(x)
    with torch.no_grad():
        temp = (rounded - x)
    rounded_through = x + temp # rounded_through = x + K.stop_gradient(rounded - x)
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


def _hard_sigmoid(x):
    '''Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return torch.clamp((x+1)/2, 0, 1)



def quantized_relu(W, nb=16):

    '''The weights' binarization function, 

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    #non_sign_bits = nb-1
    #m = pow(2,non_sign_bits)
    #Wq = K.clip(round_through(W*m),0,m-1)/m

    nb_bits = nb
    Wq = torch.clamp(2. * (round_through(_hard_sigmoid(W) * pow(2, nb_bits)) / pow(2, nb_bits)) - 1., 0,
                1 - 1.0 / pow(2, nb_bits - 1))
    return Wq

def to_hinge_cat(inp_tensor, num_class):
    #import pdb; pdb.set_trace()
    out_tensor = torch.ones([inp_tensor.shape[0], num_class], device="cuda")*-1
    out_tensor[torch.arange(inp_tensor.shape[0]).to("cuda"), torch.tensor(inp_tensor, dtype = int, device="cuda")] = 1
    #out_tensor[torch.arange(64), inp_tensor.int()] = 1
    return out_tensor


class Net(nn.Module):
    def __init__(self, quant_nb):
        super(Net, self).__init__()
        self.quant_nb = quant_nb
        self.conv1 = nn.Conv2d(1, 64, 5, 1, padding = 2, bias = False)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, padding = 2, bias = False)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.fc1 = nn.Linear(64*28*28, 10, bias=False)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        self.conv1.weight.data = quantize(self.conv1.weight.data, nb=self.quant_nb)
        x = quantized_relu(self.conv1(x), nb = self.quant_nb)
        self.conv2.weight.data = quantize(self.conv2.weight.data, nb=self.quant_nb)
        x = quantized_relu(self.conv2(x), nb = self.quant_nb)
        x = x.view(-1, 28*28*64)
        self.fc1.weight.data = quantize(self.fc1.weight.data, nb=self.quant_nb)
        x = self.fc1(x)
        return F.softmax(x)


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
        # https://stats.stackexchange.com/questions/198038/cross-entropy-or-log-likelihood-in-output-layer
        #loss = F.cross_entropy(output, target) # combines softmax + nll
        #loss = F.nll_loss(output, target)
        #import pdb; pdb.set_trace()
        target_cat_hinge = to_hinge_cat(target, 10)
        loss = squared_hinge(target_cat_hinge.to(device), output)

        loss.backward()
        optimizer.step()

        #import pdb; pdb.set_trace()

        pred = output.argmax(dim=1, keepdim=True)
        correct_guess += pred.eq(target.view_as(pred)).sum().item()
        loss_guess += loss.item()
        #if batch_idx % 100 == 0:
        print('\r Train Epoch: {} [{:.0f}%]\tLoss: {:.6f} \tAccuarcy: {:.6f} \r'.format(
        epoch,
        100. * batch_idx / len(train_loader), loss_guess/(batch_idx+1), 100* correct_guess/count ), end="")
        #batch_idx * len(data), len(train_loader.dataset)
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
            #test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            target_cat_hinge = to_hinge_cat(target, 10)
            test_loss += squared_hinge(target_cat_hinge.to(device), output)

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= count_bx

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset), test_loss


use_cuda = torch.cuda.is_available()

torch.manual_seed(69)

device = torch.device("cuda" if use_cuda else "cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()#,
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=False, num_workers = 2)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()#,
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=False, num_workers = 2)


model = Net(quant_nb = 8).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0, dampening=0, weight_decay=0, nesterov=False)#, momentum=args.momentum)

from torchsummary import summary
summary(model, (1, 28, 28))

teacc, teloss, taacc, taloss = [], [], [], []  
for epoch in range(1, 3):
    acc, lossv = train(model, device, train_loader, optimizer, epoch)
    taacc.append(acc)
    taloss.append(lossv)
    acc, lossv = test(model, device, test_loader)
    teacc.append(acc)
    teloss.append(lossv)

# import matplotlib.pyplot as plt

# #fig.set_size_inches(10, 10)
# plt.clf()
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.plot(taacc , label="Training Accuracy", color="black")
# plt.plot(teacc , label="Test Accuracy", color="blue")
# plt.legend(loc = 'best')
# plt.title("Accuarcy Pytorch")

# plt.tight_layout()
# plt.savefig("figures/torch_acc.png")


# plt.clf()
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.plot(taloss , label="Training Loss", color="black")
# plt.plot(teloss , label="Test Loss", color="blue")
# plt.legend(loc = 'best')
# plt.title("Loss Pytorch")

# plt.tight_layout()
# plt.savefig("figures/torch_loss.png")


# plt.clf()
# plt.matshow(model.conv1.weight.view([40,40]).to("cpu").detach().numpy())
# plt.title("Accuarcy Pytorch Conv1")
# plt.savefig("figures/torch_weights_conv1.png")

# plt.clf()
# plt.matshow(model.conv2.weight.view([320,320]).to("cpu").detach().numpy())
# plt.title("Accuarcy Pytorch Conv2")
# plt.savefig("figures/torch_weights_conv2.png")

# plt.clf()
# plt.matshow(model.fc1.weight.view([980,512]).to("cpu").detach().numpy())
# plt.title("Accuarcy Pytorch fc1")
# plt.savefig("figures/torch_weights_fc1.png")


#model.conv1.weight.shape
#model.conv2.weight.shape
#model.fc1.weight.shape

