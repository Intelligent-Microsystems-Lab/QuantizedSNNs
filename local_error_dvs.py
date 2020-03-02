import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
import time
import math
import numpy as np
import pandas as pd
import argparse
import datetime

import quantization
from localQ import sparse_data_generator_DVS, smoothstep, superspike, QLinearLayerSign, LIFDenseLayer, LIFConv2dLayer





def save_vid_of_input(x_temp, y_temp):
    # # visualize
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    gest_mapping = {
        0:"club",
        1:"diamond",
        2:"heart",
        3:"spade",
    }


    plt.clf()
    fig1 = plt.figure()

    ims = []
    for j in np.arange(x_local.shape[0]):
        #temp_show = downsample(x_local[:,:,:,:,j])*16
        for i in np.arange(x_local.shape[4]):

            #temp_show = downsample(x_local[:,:,:,:,i])*16

            #temp_show = torch.cat((temp_show, temp_show), dim = 1)
            temp_show = torch.cat((x_local[:,:,:,:,i], x_local[:,:,:,:,i]), dim = 1)
            mask1 = (temp_show > 0) # this might change
            mask2 = (temp_show < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            temp_show = torch.zeros_like(temp_show)
            temp_show[mask1] = 1 
            temp_show[mask2] = 1

            ims.append((plt.imshow( temp_show[j,0,:,:].cpu()), plt.text(.5, .1, gest_mapping[y_temp[j].item()], fontsize=12), ))
            
    im_ani = animation.ArtistAnimation(fig1, ims, interval=1, repeat_delay=2000, blit=True)
    im_ani.save('../dvs_poker_{date:%Y-%m-%d_%H:%M:%S}.mp4'.format( date=datetime.datetime.now()))



def sparse_data_generator_DVSPoker(X, y, batch_size, nb_steps, shuffle, device, test = False):
    number_of_batches = len(y)//batch_size
    sample_index = np.arange(len(y))
    nb_steps = nb_steps -1
    y = np.array(y)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        all_events = np.array([[],[],[],[],[],[],[]]).T


        for bc,idx in enumerate(batch_index):
            temp = np.append(np.ones((X[idx].shape[0], 1))*bc, X[idx], axis=1)
            all_events = np.append(all_events, temp, axis = 0)

        # to matrix
        all_events = all_events[:,[0,4,5,1,6]]
        sparse_matrix = torch.sparse.FloatTensor(torch.LongTensor(all_events[:,[True, True, True, True, False]].T), torch.FloatTensor(all_events[:,4])).to_dense()

        # quick trick...
        sparse_matrix[sparse_matrix < 0] = -1
        sparse_matrix[sparse_matrix > 0] = 1

        sparse_matrix = sparse_matrix.reshape(torch.Size([sparse_matrix.shape[0], 1, sparse_matrix.shape[1], sparse_matrix.shape[2], sparse_matrix.shape[3]]))

        y_batch = torch.tensor(y[batch_index], dtype = int)
        try:
            yield sparse_matrix.to(device=device), y_batch.to(device=device)
            counter += 1
        except StopIteration:
            return


verbose_output = False
ap = argparse.ArgumentParser()
ap.add_argument("-dir", "--dir", type = str, help = "output dir")
args = vars(ap.parse_args())

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float

# load data
with open('../slow_poker_500_train.pickle', 'rb') as f:
    data = pickle.load(f)
x_train = data[0]
y_train = data[1]

with open('../slow_poker_500_test.pickle', 'rb') as f:
    data = pickle.load(f)
x_test = data[0]
y_test = data[1]

# set quant level
quantization.global_wb = 8
quantization.global_ub = 8
quantization.global_qb = 8
quantization.global_pb = 8
quantization.global_gb = 8
quantization.global_eb = 8
quantization.global_rb = 16
quantization.global_lr = 1
quantization.global_beta = 1.5 #quantization.step_d(quantization.global_wb)-.5

# set parameters
ms = 1e-3
delta_t = 1*ms

T = 500*ms
T_test = 500*ms
burnin = 50*ms
batch_size = 8
output_neurons = 4

tau_mem = torch.Tensor([20*ms]).to(device)#torch.Tensor([5*ms, 35*ms]).to(device)
tau_syn = torch.Tensor([7.5*ms]).to(device)#torch.Tensor([5*ms, 10*ms]).to(device)
tau_ref = torch.Tensor([0*ms]).to(device)
thr = torch.Tensor([.4]).to(device)

lambda1 = .2 
lambda2 = .1


# construct layers
dropout_p = .99
dropout_learning = nn.Dropout(p=dropout_p)

downsample = nn.AvgPool2d(kernel_size = 4, stride = 4)

layer1 = LIFConv2dLayer(inp_shape = (2, 128, 128), kernel_size = 5, out_channels = 16, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout1 = QLinearLayerSign(np.prod(layer1.out_shape), output_neurons).to(device)

layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape, kernel_size = 5, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout2 = QLinearLayerSign(np.prod(layer2.out_shape), output_neurons).to(device)

layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape, kernel_size = 5, out_channels = 32, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, pooling = 2, padding = 2, thr = thr, device = device).to(device)
random_readout3 = QLinearLayerSign(np.prod(layer3.out_shape), output_neurons).to(device)

layer4 = LIFDenseLayer(in_channels = np.prod(layer3.out_shape), out_channels = output_neurons, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = delta_t, thr = thr, device = device).to(device)

#log_softmax_fn = nn.LogSoftmax(dim=1) # log probs for nll
#nll_loss = torch.nn.NLLLoss()
sl1_loss = torch.nn.SmoothL1Loss()

all_parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters()) + list(layer4.parameters())

# initlialize optimizier
opt = torch.optim.SGD(all_parameters, lr=1)

train_acc = []
test_acc = []


print("WPQUEG Quantization: {0}{1}{2}{3}{4}{5}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb))


for e in range(50):
    correct = 0
    total = 0
    tcorrect = 0
    ttotal = 0

    correct1_train = 0 # note over all time steps now...
    correct2_train = 0
    correct3_train = 0
    correct4_train = 0
    total_train = 0
    correct1_test = 0
    correct2_test = 0
    correct3_test = 0
    correct4_test = 0
    total_test = 0
    loss_hist = []

    start_time = time.time()

    rec_video = True
    for x_local, y_local in sparse_data_generator_DVSPoker(x_train, y_train, batch_size = batch_size, nb_steps = T / ms, shuffle = True, device = device):
        print("creating video")
        save_vid_of_input(x_local, y_local)
        print("video done")
        y_onehot = torch.Tensor(len(y_local), output_neurons).to(device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)

        class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)

        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])
        layer4.state_init(x_local.shape[0])

        # burnin
        for t in range(int(burnin/ms)):
            #down_spikes = downsample(x_local[:,:,:,:,t])*16

            # two channel trick
            #down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
            down_spikes = torch.cat((x_local[:,:,:,:,t], x_local[:,:,:,:,t]), dim = 1)
            mask1 = (down_spikes > 0) # this might change
            mask2 = (down_spikes < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            down_spikes = torch.zeros_like(down_spikes)
            down_spikes[mask1] = 1 
            down_spikes[mask2] = 1

            # same same but different
            # down_spikes[down_spikes != 0] = 1

            # bi directional


            # DTN

            out_spikes1 = layer1.forward(down_spikes)
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)

        # training
        for t in range(int(burnin/ms), int(T/ms)):
            total_train += y_local.size(0)
            loss_gen = 0
            #down_spikes = downsample(x_local[:,:,:,:,t])*16

            # two channel trick
            #down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
            down_spikes = torch.cat((x_local[:,:,:,:,t], x_local[:,:,:,:,t]), dim = 1)
            mask1 = (down_spikes > 0) # this might change
            mask2 = (down_spikes < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            down_spikes = torch.zeros_like(down_spikes)
            down_spikes[mask1] = 1 
            down_spikes[mask2] = 1

            out_spikes1 = layer1.forward(down_spikes)
            rreadout1 = random_readout1(dropout_learning(smoothstep(layer1.U.reshape([x_local.shape[0], np.prod(layer1.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct1_train += (predicted == y_local).sum().item()
            loss_gen += sl1_loss(((rreadout1 / rreadout1.abs().max())+1)*.5, y_onehot) #+ lambda1 * F.relu(layer1.U+.01).mean() + lambda2 * F.relu(thr-layer1.U).mean()

            out_spikes2 = layer2.forward(out_spikes1)
            rreadout2 = random_readout2(dropout_learning(smoothstep(layer2.U.reshape([x_local.shape[0], np.prod(layer2.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout2.data, 1)
            correct2_train += (predicted == y_local).sum().item()
            loss_gen += sl1_loss( ((rreadout2 / rreadout2.abs().max())+1)*.5, y_onehot) #+ lambda1 * F.relu(layer2.U+.01).mean() + lambda2 * F.relu(thr-layer2.U).mean()

            out_spikes3 = layer3.forward(out_spikes2)
            rreadout3 = random_readout3(dropout_learning(smoothstep(layer3.U.reshape([x_local.shape[0], np.prod(layer3.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct3_train += (predicted == y_local).sum().item()
            loss_gen += sl1_loss( ((rreadout3 / rreadout3.abs().max())+1)*.5, y_onehot) #+ lambda1 * F.relu(layer3.U+.01).mean() + lambda2 * F.relu(thr-layer3.U).mean()

            # flattening for spiking readout layer
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)
            loss_gen += sl1_loss(smoothstep(layer4.U), y_onehot)
            _, predicted = torch.max(out_spikes4, 1)
            correct4_train += (predicted == y_local).sum().item()
            #y_log_p4 = log_softmax_fn(smoothstep(layer4.U))
            #gen_loss +=  sl1_loss(smoothstep(layer4.U), y_onehot) + lambda1 * F.relu(layer4.U+.01).mean() + lambda2 * F.relu(.1-layer4.U).mean()

            loss_gen.backward()
            opt.step()
            opt.zero_grad()

            loss_hist.append(loss_gen.item())
            class_rec += out_spikes4

        correct += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        total += len(y_local)
    train_time = time.time()


    # test accuracy
    for x_local, y_local in sparse_data_generator_DVSPoker(x_test, y_test, batch_size = batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True):
        y_local = y_local -1
        class_rec = torch.zeros([x_local.shape[0], output_neurons]).to(device)
        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])
        layer4.state_init(x_local.shape[0])

        # burnin
        for t in range(int(burnin/ms)):
            #down_spikes = downsample(x_local[:,:,:,:,t])*16

            # two channel trick
            #down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
            down_spikes = torch.cat((x_local[:,:,:,:,t], x_local[:,:,:,:,t]), dim = 1)
            mask1 = (down_spikes > 0) # this might change
            mask2 = (down_spikes < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            down_spikes = torch.zeros_like(down_spikes)
            down_spikes[mask1] = 1 
            down_spikes[mask2] = 1

            out_spikes1 = layer1.forward(down_spikes)
            out_spikes2 = layer2.forward(out_spikes1)
            out_spikes3 = layer3.forward(out_spikes2)
            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)

        # testing
        for t in range(int(burnin/ms), int(T_test/ms)):
            total_test += y_local.size(0)
            #down_spikes = downsample(x_local[:,:,:,:,t])*16

            # two channel trick
            #down_spikes = torch.cat((down_spikes, down_spikes), dim = 1)
            down_spikes = torch.cat((x_local[:,:,:,:,t], x_local[:,:,:,:,t]), dim = 1)
            mask1 = (down_spikes > 0) # this might change
            mask2 = (down_spikes < 0)
            mask1[:,0,:,:] = False
            mask2[:,1,:,:] = False
            down_spikes = torch.zeros_like(down_spikes)
            down_spikes[mask1] = 1 
            down_spikes[mask2] = 1

            # dropout kept active -> decolle note
            out_spikes1 = layer1.forward(down_spikes)
            rreadout1 = random_readout1(dropout_learning(smoothstep(layer1.U.reshape([x_local.shape[0], np.prod(layer1.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct1_test += (predicted == y_local).sum().item()

            out_spikes2 = dropout_learning(layer2.forward(out_spikes1)) 
            rreadout2 = random_readout2(dropout_learning(smoothstep(layer2.U.reshape([x_local.shape[0], np.prod(layer2.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout2.data, 1)
            correct2_test += (predicted == y_local).sum().item()

            out_spikes3 = dropout_learning(layer3.forward(out_spikes2))
            rreadout3 = random_readout3(dropout_learning(smoothstep(layer3.U.reshape([x_local.shape[0], np.prod(layer3.out_shape)]))) * dropout_p)
            _, predicted = torch.max(rreadout1.data, 1)
            correct3_test += (predicted == y_local).sum().item()

            out_spikes3 = out_spikes3.reshape([x_local.shape[0], np.prod(layer3.out_shape)])
            out_spikes4 = layer4.forward(out_spikes3)
            _, predicted = torch.max(out_spikes4, 1)
            correct4_test += (predicted == y_local).sum().item()

            class_rec += out_spikes4
        tcorrect += (torch.max(class_rec, dim = 1).indices == y_local).sum() 
        ttotal += len(y_local)
    inf_time = time.time()


    correct = correct.item()
    tcorrect = tcorrect.item()
    train_acc.append(correct/total)
    test_acc.append(tcorrect/ttotal)
    #print("Epoch {0} | Loss: {1:.4f} Train Acc: {2:.4f} Test Acc: {3:.4f} Train Time: {4:.4f}s Inference Time: {5:.4f}s".format(e+1, np.mean(loss_hist), correct.item()/total, tcorrect.item()/ttotal, train_time-start_time, inf_time - train_time)) 
    #np.mean(loss_hist2), np.mean(loss_hist3), np.mean(loss_hist4),
    if verbose_output:
        print("Epoch {0} | Loss: {1:.4f}, {2:.0f}, {3:.0f}, {4:.0f} Train Acc 1: {5:.4f} Test Acc 1: {6:.4f} Train Acc 2: {7:.4f} Test Acc 2: {8:.4f} Train Acc 3: {9:.4f} Test Acc 3: {10:.4f} Train Acc 4: {11:.4f} Test Acc 4: {12:.4f}  TRAIN_ACC: {13:.4f} TEST_ACC: {14:.4f}  Train Time: {15:.4f}s Inference Time: {16:.4f}s".format(e+1, np.mean(loss_hist), -1, -1, -1, correct1_train/total_train, correct1_test/total_train, correct2_train/total_train, correct2_test/total_train, correct3_train/total_train, correct3_test/total_train, correct4_train/total_train, correct4_test/total_train, correct/total, tcorrect/ttotal, train_time-start_time, inf_time - train_time))
    else:
        print("Epoch {0} | TRAIN_ACC: {1:.4f} TEST_ACC: {2:.4f}  Train Time: {3:.4f}s Inference Time: {4:.4f}s".format(e+1, correct/total, tcorrect/ttotal, train_time-start_time, inf_time - train_time))


# saving results/weights
results = {'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer2':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer3':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'layer4':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu()], 'loss':[loss_hist]} # 'test_acc': test_acc, 'train_acc': train_acc, , 'train_idx':shuffle_idx_ta, 'test_idx':shuffle_idx_te
with open('hello.pkl', 'wb') as f:
    pickle.dump(results, f)

# Epoch 41 | Loss: 2.6689 Train Acc: 0.0816 Test Acc: 0.0833 Train Time: 734.5396s Inference Time: 298.9132s