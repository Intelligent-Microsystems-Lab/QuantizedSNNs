import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
import time
import math
import numpy as np

import quantization
from localQ import sparse_data_generator, smoothstep, superspike, QLinearLayerSign, LIFDenseLayer, LIFConv2dLayer

ap = argparse.ArgumentParser()
ap.add_argument("-dir", "--dir", type = str, help = "activation bits")
args = vars(ap.parse_args())


def run_main(xta, xte, yta, yte, resf_name, bit_change):

    shuffle_idx = torch.randperm(len(yta))
    x_train = xta[shuffle_idx]
    y_train = yta[shuffle_idx]
    shuffle_idx = torch.randperm(len(yte))
    x_test = xte[shuffle_idx]
    y_test = yte[shuffle_idx]

    # fixed subsampling
    # train: 300 samples per class -> 3000
    # test: 103 samples per class -> 1030 (a wee more than 1024)
    train_samples = 3000
    test_samples = 1030
    num_classes = 10
    index_list_train = []
    index_list_test = []
    for i in range(10):
        index_list_train.append((y_train == i).nonzero()[:int(train_samples/num_classes)])
        index_list_test.append((y_test == i).nonzero()[:int(test_samples/num_classes)])
    index_list_train = torch.cat(index_list_train).reshape([train_samples])
    index_list_test = torch.cat(index_list_test).reshape([test_samples])

    x_train = x_train[index_list_train, :]
    x_test = x_test[index_list_test, :]
    y_train = y_train[index_list_train]
    y_test = y_test[index_list_test]


    #quantization.global_beta = 1.5
    quantization.global_wb = bit_change
    quantization.global_ub = 8
    quantization.global_qb = 8
    quantization.global_pb = 8
    quantization.global_gb = 8
    quantization.global_eb = 8
    quantization.global_rb = 16
    quantization.global_lr = 1
    quantization.global_beta = 1.5 #quantization.step_d(quantization.global_wb)-.5
    # effect of global beta 

    ms = 1e-3
    delta_t = 1*ms

    T = 500*ms
    T_test = 1000*ms
    burnin = 50*ms
    batch_size = 128
    output_neurons = 10

    tau_mem = torch.Tensor([5*ms, 35*ms]).to(device)
    tau_syn = torch.Tensor([5*ms, 10*ms]).to(device)
    tau_ref = torch.Tensor([0*ms]).to(device)
    thr = torch.Tensor([.4]).to(device)

    lambda1 = .2 
    lambda2 = .1

    dropout_learning = nn.Dropout(p=.5)

    layer1 = LIFConv2dLayer(inp_shape = x_train.shape[1:], kernel_size = 7, out_channels = 16, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
    random_readout1 = QLinearLayerSign(np.prod(layer1.out_shape), output_neurons).to(device)

    layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape, kernel_size = 7, out_channels = 24, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 1, padding = 2, thr = thr, device = device).to(device)
    random_readout2 = QLinearLayerSign(np.prod(layer2.out_shape), output_neurons).to(device)

    layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape, kernel_size = 7, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
    random_readout3 = QLinearLayerSign(np.prod(layer3.out_shape), output_neurons).to(device)

    layer4 = LIFDenseLayer(in_channels = np.prod(layer3.out_shape), out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)

    log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
    nll_loss = torch.nn.NLLLoss()

    opt1 = torch.optim.SGD(layer1.parameters(), lr=1)
    opt2 = torch.optim.SGD(layer2.parameters(), lr=1)
    opt3 = torch.optim.SGD(layer3.parameters(), lr=1)
    opt4 = torch.optim.SGD(layer4.parameters(), lr=1)
    # scheduler1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=20, gamma=0.5)
    # scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=20, gamma=0.5)
    # scheduler3 = torch.optim.lr_scheduler.StepLR(opt3, step_size=20, gamma=0.5)
    # scheduler4 = torch.optim.lr_scheduler.StepLR(opt4, step_size=20, gamma=0.5)

    print("WPQUEG Quantization: {0}{1}{2}{3}{4}{5}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb))

    train_acc = []
    test_acc = []
    for e in range(1):
        correct = 0
        total = 0
        tcorrect = 0
        ttotal = 0
        loss_hist = []
        start_time = time.time()

        for x_local, y_local in sparse_data_generator(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, samples = train_samples, max_hertz = 50, shuffle = True, device = device):
            class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)

            layer1.state_init(x_local.shape[0])
            layer2.state_init(x_local.shape[0])
            layer3.state_init(x_local.shape[0])
            layer4.state_init(x_local.shape[0])

            # burnin
            for t in range(int(burnin/ms)):
                out_spikes1 = layer1.forward(x_local[:,:,:,:,t])
                out_spikes2 = layer2.forward(out_spikes1)
                out_spikes3 = layer3.forward(out_spikes2)
                out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
                out_spikes4 = layer4.forward(out_spikes3)

            # training
            for t in range(int(burnin/ms), int(T/ms)):
                out_spikes1 = layer1.forward(x_local[:,:,:,:,t])
                rreadout1 = random_readout1(dropout_learning(smoothstep(layer1.U.reshape([x_local.shape[0], np.prod(layer1.out_shape)]))))
                y_log_p1 = log_softmax_fn(rreadout1)
                loss_t1 = nll_loss(y_log_p1, y_local) + lambda1 * F.relu(layer1.U+.01).mean() + lambda2 * F.relu(thr-layer1.U).mean()
                loss_t1.backward()
                opt1.step()
                opt1.zero_grad()

                out_spikes2 = layer2.forward(out_spikes1)
                rreadout2 = random_readout2(dropout_learning(smoothstep(layer2.U.reshape([x_local.shape[0], np.prod(layer2.out_shape)]))))
                y_log_p2 = log_softmax_fn(rreadout2)
                loss_t2 = nll_loss(y_log_p2, y_local) + lambda1 * F.relu(layer2.U+.01).mean() + lambda2 * F.relu(thr-layer2.U).mean()
                loss_t2.backward()
                opt2.step()
                opt2.zero_grad()

                out_spikes3 = layer3.forward(out_spikes2)
                rreadout3 = random_readout3(dropout_learning(smoothstep(layer3.U.reshape([x_local.shape[0], np.prod(layer3.out_shape)]))))
                y_log_p3 = log_softmax_fn(rreadout3)
                loss_t3 = nll_loss(y_log_p3, y_local) + lambda1 * F.relu(layer3.U+.01).mean() + lambda2 * F.relu(thr-layer3.U).mean()
                loss_t3.backward()
                opt3.step()
                opt3.zero_grad()

                out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
                out_spikes4 = layer4.forward(out_spikes3)
                y_log_p4 = log_softmax_fn(smoothstep(layer4.U))
                loss_t4 = nll_loss(y_log_p4, y_local) + lambda1 * F.relu(layer4.U+.01).mean() + lambda2 * F.relu(.1-layer4.U).mean()
                loss_t4.backward()
                opt4.step()
                opt4.zero_grad()

                loss_hist.append(loss_t4.item())
                class_rec += out_spikes4

            correct += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
            total += len(y_local)
        train_time = time.time()


        # compute test accuracy
        for x_local, y_local in sparse_data_generator(x_test, y_test, batch_size = batch_size, nb_steps = T_test/ms, samples = test_samples, max_hertz = 50, shuffle = True, device = device):
            class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)
            layer1.state_init(x_local.shape[0])
            layer2.state_init(x_local.shape[0])
            layer3.state_init(x_local.shape[0])
            layer4.state_init(x_local.shape[0])

            for t in range(int(T_test/ms)):
                out_spikes1 = layer1.forward(x_local[:,:,:,:,t])
                out_spikes2 = layer2.forward(out_spikes1)
                out_spikes3 = layer3.forward(out_spikes2)
                out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
                out_spikes4 = layer4.forward(out_spikes3)
                class_rec += out_spikes4
            tcorrect += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
            ttotal += len(y_local)
        inf_time = time.time()

        # scheduler1.step()
        # scheduler2.step()
        # scheduler3.step()
        # scheduler4.step()

        train_acc.append(correct.item()/total)
        test_acc.append(tcorrect.item()/ttotal)
        print("Epoch {0} | Loss: {1:.4f} Train Acc: {2:.4f} Test Acc: {3:.4f} Train Time: {4:.4f}s Inference Time: {5:.4f}s".format(e+1, np.mean(loss_hist), correct.item()/total, tcorrect.item()/ttotal, train_time-start_time, inf_time - train_time)) 


    # saving results/weights
    results = {'layer1':[layer1.weights, layer1.bias], 'layer2':[layer1.weights, layer1.bias], 'layer3':[layer1.weights, layer1.bias], 'layer4':[layer1.weights, layer1.bias], 'test_acc': test_acc, 'train_acc': train_acc}
    with open(resf_name + '.pkl', 'wb') as f:
        pickle.dump(results, f)


# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float

# load data
train_dataset = torchvision.datasets.MNIST('../data', train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.MNIST('../data', train=False, transform=None, target_transform=None, download=True)

# standardize data
x_train = train_dataset.data.type(dtype)/255
x_train = x_train.reshape((x_train.shape[0],) + (1,) + x_train.shape[1:])
x_test = test_dataset.data.type(dtype)/255
x_test = x_test.reshape((x_test.shape[0],) + (1,) + x_test.shape[1:])
y_train = train_dataset.targets
y_test  = test_dataset.targets


for i in [2,3,4,5,6,7,8]:
    run_main(x_train, x_test, y_train, y_test, args['dir'] + '/wb{0}'.format(i), i)

