import pickle, argparse, time,  math, datetime, uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import quantization
import localQ
from localQ import sparse_data_generator_Static, sparse_data_generator_DVSGesture, sparse_data_generator_DVSPoker, LIFConv2dLayer, prep_input, acc_comp, create_graph, DTNLIFConv2dLayer, create_graph2


# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
dtype = torch.float32 
ms = 1e-3



parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-set", type=str, default="Gesture", help='Input date set: Poker/Gesture')

parser.add_argument("--global_wb", type=int, default=8, help='Weight bitwidth')
parser.add_argument("--global_qb", type=int, default=10, help='Synapse bitwidth')
parser.add_argument("--global_pb", type=int, default=12, help='Membrane trace bitwidth')
parser.add_argument("--global_rfb", type=int, default=2, help='Refractory bitwidth')

parser.add_argument("--global_sb", type=int, default=6, help='Learning signal bitwidth')
parser.add_argument("--global_gb", type=int, default=10, help='Gradient bitwidth')
parser.add_argument("--global_eb", type=int, default=6, help='Error bitwidth')

parser.add_argument("--global_ub", type=int, default=6, help='Membrane Potential bitwidth')
parser.add_argument("--global_ab", type=int, default=6, help='Activation bitwidth')
parser.add_argument("--global_sig", type=int, default=6, help='Sigmoid bitwidth')

parser.add_argument("--global_rb", type=int, default=16, help='Gradient RNG bitwidth')
parser.add_argument("--global_lr", type=int, default=1, help='Learning rate for quantized gradients')
parser.add_argument("--global_lr_sgd", type=float, default=1.0e-9, help='Learning rate for SGD')
parser.add_argument("--global_beta", type=float, default=1.5, help='Beta for weight init')

parser.add_argument("--delta_t", type=float, default=1*ms, help='Time step in ms')
parser.add_argument("--input_mode", type=int, default=0, help='Spike processing method')
parser.add_argument("--ds", type=int, default=4, help='Downsampling')
parser.add_argument("--epochs", type=int, default=320, help='Epochs for training')
parser.add_argument("--lr_div", type=int, default=80, help='Learning rate divide interval')
parser.add_argument("--batch_size", type=int, default=72, help='Batch size')

parser.add_argument("--PQ_cap", type=float, default=1, help='Value cap for membrane and synpase trace')
parser.add_argument("--weight_mult", type=float, default=4e-5, help='Weight multiplier')
parser.add_argument("--dropout_p", type=float, default=.5, help='Dropout probability')
parser.add_argument("--lc_ampl", type=float, default=.5, help='Magnitude amplifier for weight init')
parser.add_argument("--l1", type=float, default=.001, help='Regularizer 1')
parser.add_argument("--l2", type=float, default=.001, help='Regularizer 2')

parser.add_argument("--tau_mem_lower", type=float, default=5, help='Tau mem lower bound')
parser.add_argument("--tau_mem_upper", type=float, default=35, help='Tau mem upper bound')
parser.add_argument("--tau_syn_lower", type=float, default=5, help='Tau syn lower bound')
parser.add_argument("--tau_syn_upper", type=float, default=10, help='Tau syn upper bound')
parser.add_argument("--tau_ref", type=float, default=1/.35, help='Tau ref')


args = parser.parse_args()


# set quant level
quantization.global_wb  = args.global_wb
quantization.global_qb  = args.global_qb
quantization.global_pb  = args.global_pb
quantization.global_rfb = args.global_rfb

quantization.global_sb  = args.global_sb
quantization.global_gb  = args.global_gb
quantization.global_eb  = args.global_eb

quantization.global_ub  = args.global_ub
quantization.global_ab  = args.global_ab
quantization.global_sig = args.global_sig

quantization.global_rb = args.global_rb
quantization.global_lr = args.global_lr
quantization.global_lr_sgd = args.global_lr_sgd
quantization.global_beta = args.global_beta
quantization.weight_mult = args.weight_mult

localQ.lc_ampl = args.lc_ampl

tau_mem = torch.tensor([args.tau_mem_lower*ms, args.tau_mem_upper*ms], dtype = dtype).to(device)
tau_ref = torch.tensor([args.tau_ref*ms], dtype = dtype).to(device)
tau_syn = torch.tensor([args.tau_syn_lower*ms, args.tau_syn_upper*ms], dtype = dtype).to(device) 


if args.data_set == "Poker":
    ds_name = "DVS Poker"
    with open('data/slow_poker_500_train.pickle', 'rb') as f:
        data = pickle.load(f)
    x_train = data[0].tolist()
    for i in range(len(x_train)):
        x_train[i] = x_train[i][:,[0,3,4,5]]
        x_train[i][:,3][x_train[i][:,3] == -1] = 0
        x_train[i] = x_train[i].astype('uint32')
    y_train = data[1]

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    idx_temp = np.arange(len(x_train))
    np.random.shuffle(idx_temp)
    idx_train = idx_temp[0:int(len(y_train)*.8)]
    idx_val = idx_temp[int(len(y_train)*.8):]

    x_train, x_val = x_train[idx_train], x_train[idx_val]
    y_train, y_val = y_train[idx_train], y_train[idx_val]

    with open('data/slow_poker_500_test.pickle', 'rb') as f:
        data = pickle.load(f)
    x_test = data[0].tolist()
    for i in range(len(x_test)):
        x_test[i] = x_test[i][:,[0,3,4,5]]
        x_test[i][:,3][x_test[i][:,3] == -1] = 0
        x_test[i] = x_test[i].astype('uint32')
    y_test = data[1]

    output_neurons = 4
    T = 500*ms
    T_test = 500*ms
    burnin = 50*ms
    x_size = 32
    y_size = 32
    train_tflag = True



elif args.data_set == "Gesture":
    ds_name = "DVS Gesture"
    with open('data/train_dvs_gesture88.pickle', 'rb') as f:
        data = pickle.load(f)
    x_train = np.array(data[0])
    y_train = np.array(data[1], dtype = int) - 1

    idx_temp = np.arange(len(x_train))
    np.random.shuffle(idx_temp)
    idx_train = idx_temp[0:int(len(y_train)*.8)]
    idx_val = idx_temp[int(len(y_train)*.8):]

    x_train, x_val = x_train[idx_train], x_train[idx_val]
    y_train, y_val = y_train[idx_train], y_train[idx_val]


    with open('data/test_dvs_gesture88.pickle', 'rb') as f:
        data = pickle.load(f)
    x_test = data[0]
    y_test = np.array(data[1], dtype = int) - 1

    output_neurons = 11
    T = 500*ms
    T_test = 1800*ms
    burnin = 50*ms
    x_size = 32
    y_size = 32
    train_tflag = False
else:
    raise Exception("Data set unknown.")

sl1_loss = torch.nn.MSELoss()

thr = torch.tensor([.0], dtype = dtype).to(device)
layer1 = LIFConv2dLayer(inp_shape = (2, x_size, y_size), kernel_size = 7, out_channels = 64, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = args.delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = args.dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = args.l1, l2 = args.l2, PQ_cap = args.PQ_cap, weight_mult = args.weight_mult, dtype = dtype).to(device)

layer2 = LIFConv2dLayer(inp_shape = layer1.out_shape2, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = args.delta_t, pooling = 1, padding = 2, thr = thr, device = device, dropout_p = args.dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = args.l1, l2 = args.l2, PQ_cap = args.PQ_cap, weight_mult = args.weight_mult, dtype = dtype).to(device)

layer3 = LIFConv2dLayer(inp_shape = layer2.out_shape2, kernel_size = 7, out_channels = 128, tau_mem = tau_mem, tau_syn = tau_syn, tau_ref = tau_ref, delta_t = args.delta_t, pooling = 2, padding = 2, thr = thr, device = device, dropout_p = args.dropout_p, output_neurons = output_neurons, loss_fn = sl1_loss, l1 = args.l1, l2 = args.l2, PQ_cap = args.PQ_cap, weight_mult = args.weight_mult, dtype = dtype).to(device)


all_parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters())

# initlialize optimizier
if quantization.global_gb is not None:
    opt = torch.optim.SGD(all_parameters, lr = 1)
else:
    opt = torch.optim.SGD(all_parameters, lr = quantization.global_lr_sgd)

def eval_test():
    batch_corr = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [], 'loss':[], 'act_train1':0, 'act_train2':0, 'act_train3':0, 'act_test1':0, 'act_test2':0, 'act_test3':0, 'w1u':0, 'w2u':0, 'w3u':0}
    # test accuracy
    for x_local, y_local in sparse_data_generator_DVSGesture(x_test, y_test, batch_size = args.batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True, ds = args.ds, x_size = x_size, y_size = y_size):
        rread_hist1_test = []
        rread_hist2_test = []
        rread_hist3_test = []

        y_onehot = torch.Tensor(len(y_local), output_neurons).to(device).type(dtype)
        y_onehot.zero_()
        y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)


        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])

        for t in range(int(T_test/ms)):
            test_flag = (t > int(burnin/ms))

            out_spikes1, temp_loss1, temp_corr1, _ = layer1.forward(prep_input(x_local[:,:,:,:,t], args.input_mode), y_onehot, test_flag = test_flag)
            out_spikes2, temp_loss2, temp_corr2, _ = layer2.forward(out_spikes1, y_onehot, test_flag = test_flag)
            out_spikes3, temp_loss3, temp_corr3, _ = layer3.forward(out_spikes2, y_onehot, test_flag = test_flag)

            if test_flag:
                rread_hist1_test.append(temp_corr1)
                rread_hist2_test.append(temp_corr2)
                rread_hist3_test.append(temp_corr3)


        batch_corr['test1'].append(acc_comp(rread_hist1_test, y_local, True))
        batch_corr['test2'].append(acc_comp(rread_hist2_test, y_local, True))
        batch_corr['test3'].append(acc_comp(rread_hist3_test, y_local, True))

    return torch.cat(batch_corr['test3']).mean()


w1, w2, w3, b1, b2, b3 = None, None, None, None, None, None

diff_layers_acc = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [], 'loss':[], 'act_train1':[], 'act_train2':[], 'act_train3':[], 'act_test1':[], 'act_test2':[], 'act_test3':[], 'w1update':[], 'w2update':[], 'w3update':[]}
print("WUPQR SASigEG Quantization: {0}{1}{2}{3}{4} {5}{6}{7}{8}{9} l1 {10:.3f} l2 {11:.3f} Inp {12} LR {13} Drop {14} Cap {15} thr {16}".format(quantization.global_wb, quantization.global_ub, quantization.global_pb, quantization.global_qb, quantization.global_rfb, quantization.global_sb, quantization.global_ab, quantization.global_sig, quantization.global_eb, quantization.global_gb, args.l1, args.l2, args.input_mode, quantization.global_lr if quantization.global_lr != None else quantization.global_lr_sgd, args.dropout_p, args.PQ_cap, thr.item()))
plot_file_name = "DVS_WPQUEG{0}{1}{2}{3}{4}{5}{6}_Inp{7}_LR{8}_Drop{9}_thr{10}".format(quantization.global_wb, quantization.global_pb, quantization.global_qb, quantization.global_ub, quantization.global_eb, quantization.global_gb, quantization.global_sb, args.input_mode, quantization.global_lr, args.dropout_p, thr.item())+datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
print("Epoch Loss      Train1 Train2 Train3 Test1  Test2  Test3  | TrainT   TestT")

best_vali = torch.tensor(0, device = device)

for e in range(args.epochs):
    if ((e+1)%args.lr_div)==0:
        if quantization.global_gb is not None:
            quantization.global_lr /= 2
        else:
            opt.param_groups[-1]['lr'] /= 5


    batch_corr = {'train1': [], 'test1': [],'train2': [], 'test2': [],'train3': [], 'test3': [], 'loss':[], 'act_train1':0, 'act_train2':0, 'act_train3':0, 'act_test1':0, 'act_test2':0, 'act_test3':0, 'w1u':0, 'w2u':0, 'w3u':0}
    quantization.global_w1update = 0 
    quantization.global_w2update = 0 
    quantization.global_w3update = 0 
    start_time = time.time()

    # training
    for x_local, y_local in sparse_data_generator_DVSGesture(x_train, y_train, batch_size = args.batch_size, nb_steps = T / ms, shuffle = True, test = train_tflag, device = device, ds = args.ds, x_size = x_size, y_size = y_size):

        y_onehot = torch.Tensor(len(y_local), output_neurons).to(device).type(dtype)
        y_onehot.zero_()
        y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)

        rread_hist1_train = []
        rread_hist2_train = []
        rread_hist3_train = []
        loss_hist = []


        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])

        for t in range(int(T/ms)):
            train_flag = (t > int(burnin/ms))

            out_spikes1, temp_loss1, temp_corr1, lparts1 = layer1.forward(prep_input(x_local[:,:,:,:,t], args.input_mode), y_onehot, train_flag = train_flag)
            out_spikes2, temp_loss2, temp_corr2, lparts2 = layer2.forward(out_spikes1, y_onehot, train_flag = train_flag)
            out_spikes3, temp_loss3, temp_corr3, lparts3 = layer3.forward(out_spikes2, y_onehot, train_flag = train_flag)
            


            if train_flag:
                loss_gen = temp_loss1 + temp_loss2 + temp_loss3

                loss_gen.backward()
                opt.step()
                opt.zero_grad()

                loss_hist.append(loss_gen.item())
                rread_hist1_train.append(temp_corr1)
                rread_hist2_train.append(temp_corr2)
                rread_hist3_train.append(temp_corr3)


            batch_corr['act_train1'] += int(out_spikes1.sum())
            batch_corr['act_train2'] += int(out_spikes2.sum())
            batch_corr['act_train3'] += int(out_spikes3.sum())


        batch_corr['train1'].append(acc_comp(rread_hist1_train, y_local, True))
        batch_corr['train2'].append(acc_comp(rread_hist2_train, y_local, True))
        batch_corr['train3'].append(acc_comp(rread_hist3_train, y_local, True))
        del x_local, y_local, y_onehot


    train_time = time.time()

    diff_layers_acc['train1'].append(torch.cat(batch_corr['train1']).mean())
    diff_layers_acc['train2'].append(torch.cat(batch_corr['train2']).mean())
    diff_layers_acc['train3'].append(torch.cat(batch_corr['train3']).mean())
    diff_layers_acc['act_train1'].append(batch_corr['act_train1'])
    diff_layers_acc['act_train2'].append(batch_corr['act_train2'])
    diff_layers_acc['act_train3'].append(batch_corr['act_train3'])
    diff_layers_acc['loss'].append(np.mean(loss_hist)/3)
    diff_layers_acc['w1update'].append(quantization.global_w1update)
    diff_layers_acc['w2update'].append(quantization.global_w2update)
    diff_layers_acc['w3update'].append(quantization.global_w3update)
        
    
    # test accuracy
    for x_local, y_local in sparse_data_generator_DVSGesture(x_val, y_val, batch_size = args.batch_size, nb_steps = T_test / ms, shuffle = True, device = device, test = True, ds = args.ds, x_size = x_size, y_size = y_size):
        rread_hist1_test = []
        rread_hist2_test = []
        rread_hist3_test = []

        y_onehot = torch.Tensor(len(y_local), output_neurons).to(device).type(dtype)
        y_onehot.zero_()
        y_onehot.scatter_(1, y_local.reshape([y_local.shape[0],1]), 1)


        layer1.state_init(x_local.shape[0])
        layer2.state_init(x_local.shape[0])
        layer3.state_init(x_local.shape[0])

        for t in range(int(T_test/ms)):
            test_flag = (t > int(burnin/ms))

            out_spikes1, temp_loss1, temp_corr1, _ = layer1.forward(prep_input(x_local[:,:,:,:,t], args.input_mode), y_onehot, test_flag = test_flag)
            out_spikes2, temp_loss2, temp_corr2, _ = layer2.forward(out_spikes1, y_onehot, test_flag = test_flag)
            out_spikes3, temp_loss3, temp_corr3, _ = layer3.forward(out_spikes2, y_onehot, test_flag = test_flag)

            if test_flag:
                rread_hist1_test.append(temp_corr1)
                rread_hist2_test.append(temp_corr2)
                rread_hist3_test.append(temp_corr3)

            batch_corr['act_test1'] += int(out_spikes1.sum())
            batch_corr['act_test2'] += int(out_spikes2.sum())
            batch_corr['act_test3'] += int(out_spikes3.sum())

        batch_corr['test1'].append(acc_comp(rread_hist1_test, y_local, True))
        batch_corr['test2'].append(acc_comp(rread_hist2_test, y_local, True))
        batch_corr['test3'].append(acc_comp(rread_hist3_test, y_local, True))
        del x_local, y_local, y_onehot

    inf_time = time.time()

    if best_vali.item() < torch.cat(batch_corr['test3']).mean().item():
        best_vali = torch.cat(batch_corr['test3']).mean()
        test_acc_best_vali = eval_test()
        w1 = layer1.weights.data.detach().cpu()
        w2 = layer2.weights.data.detach().cpu()
        w3 = layer3.weights.data.detach().cpu()
        b1 = layer1.bias.data.detach().cpu()
        b2 = layer2.bias.data.detach().cpu()
        b3 = layer3.bias.data.detach().cpu()

    diff_layers_acc['test1'].append(torch.cat(batch_corr['test1']).mean())
    diff_layers_acc['test2'].append(torch.cat(batch_corr['test2']).mean())
    diff_layers_acc['test3'].append(torch.cat(batch_corr['test3']).mean())
    diff_layers_acc['act_test1'].append(batch_corr['act_test1'])
    diff_layers_acc['act_test2'].append(batch_corr['act_test2'])
    diff_layers_acc['act_test3'].append(batch_corr['act_test3'])

    print("{0:02d}    {1:.3E} {2:.4f} {3:.4f} {4:.4f} {5:.4f} {6:.4f} {7:.4f} | {8:.4f} {9:.4f}".format(e+1, diff_layers_acc['loss'][-1], diff_layers_acc['train1'][-1], diff_layers_acc['train2'][-1], diff_layers_acc['train3'][-1], diff_layers_acc['test1'][-1], diff_layers_acc['test2'][-1], diff_layers_acc['test3'][-1], train_time - start_time, inf_time - train_time))
    create_graph(plot_file_name, diff_layers_acc, ds_name, test_acc_best_vali)



    # saving results and weights
    results = {
    'layer1':[layer1.weights.detach().cpu(), layer1.bias.detach().cpu(), w1, b1, layer1.sign_random_readout.weights.detach().cpu(), layer1.sign_random_readout.weight_fa.detach().cpu(), layer1.tau_mem.cpu(), layer1.tau_syn.cpu(), layer1.tau_ref.cpu()], 
    'layer2':[layer2.weights.detach().cpu(), layer2.bias.detach().cpu(), w2, b2, layer2.sign_random_readout.weights.detach().cpu(), layer2.sign_random_readout.weight_fa.detach().cpu(), layer2.tau_mem.cpu(), layer2.tau_syn.cpu(), layer2.tau_ref.cpu()], 
    'layer3':[layer3.weights.detach().cpu(), layer3.bias.detach().cpu(), w3, b3, layer3.sign_random_readout.weights.detach().cpu(), layer3.sign_random_readout.weight_fa.detach().cpu(), layer3.tau_mem.cpu(), layer3.tau_syn.cpu(), layer3.tau_ref.cpu()], 
    'acc': diff_layers_acc, 'fname':plot_file_name, 'args': args, 'evaled_test':test_acc_best_vali}
    with open('results/'+plot_file_name+'.pkl', 'wb') as f:
        pickle.dump(results, f)

