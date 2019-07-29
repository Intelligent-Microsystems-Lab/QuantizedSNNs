import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from quantization import quantize


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor()])


# MNIST
train_dataset = torchvision.datasets.MNIST('../data/MNIST', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512,
                                          shuffle=True, num_workers=2)
test_dataset = torchvision.datasets.MNIST('../data/MNIST', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=512,
                                          shuffle=True, num_workers=2)


# FMNIST
# train_dataset = torchvision.datasets.FashionMNIST('../data/MNIST', train=True, transform=transform, download=True)
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512,
#                                           shuffle=True, num_workers=2)
# test_dataset = torchvision.datasets.FashionMNIST('../data/MNIST', train=False, transform=transform, download=True)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=512,
#                                           shuffle=True, num_workers=2)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 800)
        torch.nn.init.orthogonal_(self.fc1.weight)
        self.fc2 = nn.Linear(800, 10)
        torch.nn.init.orthogonal_(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 28*28)
        with torch.no_grad():
            self.fc1.weight.data = quantize(self.fc1.weight.data, nb=8)
        x = torch.nn.functional.relu(self.fc1(x))
        with torch.no_grad():
            self.fc2.weight.data = quantize(self.fc2.weight.data, nb=8)
        x = torch.nn.functional.relu(self.fc2(x))
        return x

def get_accuracy(loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


net = Net().to(device)
m = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5.58189e-03)


for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fn(m(outputs), labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.item()

        #with torch.no_grad():
        #    net.fc1.weight.data = quantize(net.fc1.weight.data, nb=4)
        #    net.fc2.weight.data = quantize(net.fc2.weight.data, nb=4)
    print("Epoch %d, Train %.4f, Test %.4f" % (epoch, get_accuracy(trainloader, net), get_accuracy(testloader, net)))




