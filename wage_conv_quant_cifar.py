import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse

import quantization
from quantization import clee_conv2d, clee_LinearFunction, quant_act, init_layer_weights, SSE, to_cat, clip


ap = argparse.ArgumentParser()
ap.add_argument("-ab", "--ab", type = int, help = "activation bits")
ap.add_argument("-wb", "--wb", type = int, help = "weight bits")
ap.add_argument("-eb", "--eb", type = int, help="weight bits")
ap.add_argument("-gb", "--gb", type = int, help="gradient bits")
#ap.add_argument("-rb", "--random", type = int, help="random bits")
args = vars(ap.parse_args())


quantization.global_wb = args['wb']
quantization.global_ab = args['ab']
quantization.global_gb = args['gb']
quantization.global_eb = args['eb']
quantization.global_rb = 16

quantization.global_beta = 1.5
quantization.global_lr = 8

class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()

        self.device = device

        self.padder1 = nn.ZeroPad2d(1)

        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv1.scale = init_layer_weights(self.conv1.weight, self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.in_channels).to(device)

        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv2.scale = init_layer_weights(self.conv2.weight, self.conv2.kernel_size[0] * self.conv2.kernel_size[1] * self.conv2.in_channels).to(device)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv3.scale = init_layer_weights(self.conv3.weight, self.conv3.kernel_size[0] * self.conv3.kernel_size[1] * self.conv3.in_channels).to(device)

        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv4.scale = init_layer_weights(self.conv4.weight, self.conv4.kernel_size[0] * self.conv4.kernel_size[1] * self.conv4.in_channels).to(device)

        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv5.scale = init_layer_weights(self.conv5.weight, self.conv5.kernel_size[0] * self.conv5.kernel_size[1] * self.conv5.in_channels).to(device)

        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv6.scale = init_layer_weights(self.conv6.weight, self.conv6.kernel_size[0] * self.conv6.kernel_size[1] * self.conv6.in_channels).to(device)

        self.fc1 = nn.Linear(8192, 1024, bias=False)
        self.fc1.scale = init_layer_weights(self.fc1.weight, self.fc1.in_features).to(device)

        self.fc2 = nn.Linear(1024, 10, bias=False)
        self.fc2.scale = init_layer_weights(self.fc2.weight, self.fc2.in_features).to(device)


    def forward(self, x):
        batch_size = x.shape[0]

        # clip weights
        with torch.no_grad():
            self.conv1.weight.data = clip(self.conv1.weight.data, quantization.global_wb)
            self.conv2.weight.data = clip(self.conv2.weight.data, quantization.global_wb)
            self.conv3.weight.data = clip(self.conv3.weight.data, quantization.global_wb)
            self.conv4.weight.data = clip(self.conv4.weight.data, quantization.global_wb)
            self.conv5.weight.data = clip(self.conv5.weight.data, quantization.global_wb)
            self.conv6.weight.data = clip(self.conv6.weight.data, quantization.global_wb)
            self.fc1.weight.data = clip(self.fc1.weight.data, quantization.global_wb)
            self.fc2.weight.data = clip(self.fc2.weight.data, quantization.global_wb)

        # normalizing data between -1 and 1
        x = x * 255
        x = x / 127.5 - 1

        x, _ = quant_act(x)

        # conv1 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, self.conv1.weight, self.conv1.scale, True, True, False)

        # conv2 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, self.conv2.weight, self.conv2.scale, True, True, True)

        # conv3 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, self.conv3.weight, self.conv3.scale, True, True, False)

        # conv4 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, self.conv4.weight, self.conv4.scale, True, True, True)

        # conv5 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, self.conv5.weight, self.conv5.scale, True, True, False)

        # conv6 layer
        x = self.padder1(x)
        x = clee_conv2d.apply(x, self.conv6.weight, self.conv6.scale, True, True, True)
        
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

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomCrop([40, 40], pad_if_needed=True),
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomCrop([32,32]),
                       transforms.ToTensor()
                   ])),
    batch_size=64, shuffle=True, num_workers = 2)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor()
                   ])),
    batch_size=64, shuffle=False, num_workers = 2)

# setting up the model
model = Net(device).to(device)
optimizer = optim.SGD(model.parameters(), lr=1, momentum=0, dampening=0, weight_decay=0, nesterov=False)

# train
teacc, teloss, taacc, taloss = [], [], [], []  
for epoch in range(300):
    # learning rate scheduler
    if (epoch == 200) or (epoch == 250):
       quantization.global_lr /= 8
    acc, lossv = train(model, device, train_loader, optimizer, epoch)
    taacc.append(acc)
    taloss.append(lossv)
    acc, lossv = test(model, device, test_loader)
    teacc.append(acc)
    teloss.append(lossv)


# graph results
bit_string = str(quantization.global_wb) + str(quantization.global_ab) + str(quantization.global_gb) + str(quantization.global_eb)

plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.plot(taacc , label="Training Accuracy", color="black")
plt.plot(teacc , label="Test Accuracy", color="blue")
plt.legend(loc = 'best')
plt.title("Accuarcy Pytorch WAGE Acc CIFAR10 ("+bit_string+")")

plt.tight_layout()
plt.savefig("figures/torch_wage_acc_cifar10_"+bit_string+".png")

plt.clf()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(taloss , label="Training Loss", color="black")
plt.plot(teloss , label="Test Loss", color="blue")
plt.legend(loc = 'best')
plt.title("Loss Pytorch WAGE Loss CIFAR10 ("+bit_string+")")

plt.tight_layout()
plt.savefig("figures/torch_wage_loss_cifar10_"+bit_string+".png")
