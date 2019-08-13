import pickle
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from quantization import quantize


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [transforms.ToTensor()])


# MNIST
# train_dataset = torchvision.datasets.MNIST('../data/MNIST', train=True, transform=transform, download=True)
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512,
#                                           shuffle=True, num_workers=2)
# test_dataset = torchvision.datasets.MNIST('../data/MNIST', train=False, transform=transform, download=True)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=512,
#                                           shuffle=True, num_workers=2)

# CIFAR10
train_dataset = torchvision.datasets.CIFAR10('../data/CIFAR10', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512,
                                          shuffle=True, num_workers=2)
test_dataset = torchvision.datasets.CIFAR10('../data/CIFAR10', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=512,
                                          shuffle=True, num_workers=2)


# FMNIST
# train_dataset = torchvision.datasets.FashionMNIST('../data/FMNIST', train=True, transform=transform, download=True)
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512,
#                                           shuffle=True, num_workers=2)
# test_dataset = torchvision.datasets.FashionMNIST('../data/FMNIST', train=False, transform=transform, download=True)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=512,
#                                           shuffle=True, num_workers=2)



class Net(nn.Module):
    def __init__(self, quant_nb=16):
        super(Net, self).__init__()

        self.quant_nb = quant_nb

        self.fc1 = nn.Linear(32*32, 800)
        torch.nn.init.orthogonal_(self.fc1.weight)
        self.fc2 = nn.Linear(800, 10)
        torch.nn.init.orthogonal_(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 28*28)
        with torch.no_grad():
            self.fc1.weight.data = quantize(self.fc1.weight.data, nb=self.quant_nb)
            x = quantize(x, nb=self.quant_nb)
        x = torch.nn.functional.relu(self.fc1(x))
        with torch.no_grad():
            self.fc2.weight.data = quantize(self.fc2.weight.data, nb=self.quant_nb)
            x = quantize(x, nb=self.quant_nb)
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


def train_nn(net, epochs):

    m = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5.58189e-03)


    train_acc = []
    test_acc  = []
    for epoch in range(epochs):  # loop over the dataset multiple times
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

        train_acc.append(get_accuracy(trainloader, net))
        test_acc.append(get_accuracy(testloader, net))
        print("Epoch %d, Train %.4f, Test %.4f" % (epoch, train_acc[-1], test_acc[-1]))

    return train_acc, test_acc

net = Net(quant_nb = 8).to(device)
train_acc, test_acc = train_nn(net, 500)




record_train = []
record_test = []
for nbits in range(16):
    print(nbits)
    record_train.append([])
    record_test.append([])
    for j in range(5):
        net = Net(quant_nb = nbits).to(device)
        train_acc, test_acc = train_nn(net, 500)

        record_test[-1].append(test_acc)
        record_train[-1].append(train_acc)


results = {'train':record_train, 'test': record_test}

with open('results_big_run.pkl', 'wb') as f:
    pickle.dump(results, f)

#with open('results_big_run.pkl', 'rb') as f:
#    results = pickle.load(f)

#record_train, record_test = results['train'], results['test']


# test_fin = []
# for i in record_test:
#     test_fin.append(i[-1])

# train_fin = []
# for i in record_train:
#     train_fin.append(i[-1])


# plt.clf()
# plt.ylabel('Accuracy')
# plt.xlabel('Bits used for Quantization')
# plt.plot(list(np.arange(13)+3), train_fin, label="Training Accuracy", color="black")
# plt.plot(list(np.arange(13)+3), test_fin, label="Test Accuracy", color="blue")
# plt.legend(loc = 'best')
# plt.title("Effect of Bits used for Quantization")

# plt.tight_layout()
# plt.savefig("figures/effect_nbits.png")


# plt.clf()
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.plot(record_train[8], label="Training Accuracy", color="black")
# plt.plot(record_test[8], label="Test Accuracy", color="blue")
# plt.legend()
# plt.title("Quantized MNIST 800 hidden ReLu")

# plt.tight_layout()

# plt.savefig("figures/learning10nb.png")


test_rec = []
y = []
yerr = []
for i in results['test']:
    temp = []
    for j in i:
        temp.append(max(j))
    y.append(np.mean(temp))
    yerr.append(np.std(temp))

x = np.arange(len(y))

plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('NBits')

plt.errorbar(x, y , yerr=yerr, label='test')

#plt.legend(loc = 'best')
plt.title("Effect of Quantization")

plt.tight_layout()
plt.savefig("big_errorbar_500.png")



