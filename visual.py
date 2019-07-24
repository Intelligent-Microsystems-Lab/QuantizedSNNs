import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch

def mnist_train_curve(loss_hist, train_acc, test_acc, fig_title):
    plt.clf()
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(loss_hist, label="Surrogate gradient", color="red")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  

    ax2.set_ylabel('Accuracy')
    ax2.plot(train_acc, label="Training Accuracy", color="black")
    ax2.plot(test_acc, label="Test Accuracy", color="blue")
    ax2.tick_params(axis='y')

    plt.legend()
    plt.title(fig_title)

    fig.tight_layout()

    plt.savefig("figures/training_mnist_LIF"+str('{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() ))+".png")




def learning_para(log_lrs, local_loss, losses, fig_title, var_name):
    fig = plt.figure(figsize=(10,6))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(log_lrs, local_loss)
    plt.ylabel("Loss")
    plt.xlabel(var_name)
    plt.subplot(1, 2, 2)
    plt.plot(log_lrs, losses)
    plt.ylabel("Smoothed Loss")
    plt.xlabel(var_name)
    fig.tight_layout()
    plt.savefig("figures/"+fig_title+'_'+var_name+'_'+str('{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() ))+".png",  dpi=100)

def precise_figs(y_train, y_pred, local_loss, x_train):
    y_cor_train = np.where(y_train[0,:,:].cpu().detach().numpy() == 1)[0]
    x_cor_train = np.where(y_train[0,:,:].cpu().detach().numpy() == 1)[1]
    x_cor_train += 10

    y_cor_res = np.where(y_pred[0,:,:].cpu().detach().numpy() == 1)[0]
    x_cor_res = np.where(y_pred[0,:,:].cpu().detach().numpy() == 1)[1]
    x_cor_res += 10

    plt.clf()
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3,9))

    #axes[0].text(0.05, 0.95, "tau_mem: " +str(args_snn['tau_mem'])+ "\ntau_syn: "+ str(args_snn['tau_syn']) + "\ntau_vr: " + str(tau_vr) + "\nlr: " + str(lr), fontsize=9, verticalalignment='top')
    #axes[0].axis('off')
    axes[0].plot(np.log(local_loss))
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[1].imshow(x_train[0,:,:].cpu().detach().numpy().transpose(), cmap='Greys',  interpolation='nearest', origin="lower")
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Neuron')


    axes[2].scatter(y_cor_train, x_cor_train, s=7, color="blue", label="Target")
    axes[2].scatter(y_cor_res, x_cor_res, s=3, color="orange", label="Learned")
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Neuron')

    fig.tight_layout()
    plt.savefig("figures/precise_"+str('{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() ))+".png",  dpi=300)




def neuron_test(args):
    dtype = torch.float
    nb_inputs  = 1
    nb_hidden  = 1
    nb_outputs = 1


    w1 = torch.ones((nb_inputs, nb_hidden),  device=args['device'], dtype=dtype, requires_grad=True)

    w2 = torch.ones((nb_hidden, nb_outputs), device=args['device'], dtype=dtype, requires_grad=True)

    weights = [w1,w2]
    # spikes every 50 time steps
    a = torch.zeros(1, 1000, 1, device=args['device'], dtype=dtype)
    a[0,::50,0] = 3

    # step current
    c = torch.zeros(1, 1000, 1, device=args['device'], dtype=dtype)
    c[0,100:-100,0] = 1

    # bernoulli time steps
    b = torch.empty(1, 1000, 1, device=args['device'], dtype=dtype).uniform_(0, 0.1)
    b = torch.bernoulli(b)*3

    if "adex_LIF" in args['neuron_type'].__name__:  
        other_a = args['neuron_type'](inputs=a, weights=w1, tau_syn=args['tau_syn'][0][0], tau_mem=args['tau_mem'][0][0], tau_cur=args['tau_cur'][0][0], sharpness=args['sharpness'][0][0], device=args['device'], spike_fn=args['spike_fn'], a_cur=args['a_cur'][0][0], b_cur=args['b_cur'][0][0], theta=args['theta'][0][0], time_step=args['time_step'])
        other_b = args['neuron_type'](inputs=b, weights=w1, tau_syn=args['tau_syn'][0][0], tau_mem=args['tau_mem'][0][0], tau_cur=args['tau_cur'][0][0], sharpness=args['sharpness'][0][0], device=args['device'], spike_fn=args['spike_fn'], a_cur=args['a_cur'][0][0], b_cur=args['b_cur'][0][0], theta=args['theta'][0][0], time_step=args['time_step'])
        other_c = args['neuron_type'](inputs=c, weights=w1, tau_syn=args['tau_syn'][0][0], tau_mem=args['tau_mem'][0][0], tau_cur=args['tau_cur'][0][0], sharpness=args['sharpness'][0][0], device=args['device'], spike_fn=args['spike_fn'], a_cur=args['a_cur'][0][0], b_cur=args['b_cur'][0][0], theta=args['theta'][0][0], time_step=args['time_step'])
    elif "ferro_neuron" in args['neuron_type'].__name__:  
        other_a = args['neuron_type'](inputs = a, weights = w1, v_rest_e = args['v_rest_e'][0][0], v_reset_e = args['v_reset_e'][0][0], v_thresh_e = args['v_thresh_e'][0][0], refrac_e = args['refrac_e'][0][0], tau_v = args['tau_v'][0][0], del_theta = args['del_theta'][0][0], ge_max = args['ge_max'][0][0], gi_max = args['gi_max'][0][0], tau_ge = args['tau_ge'][0][0], tau_gi = args['tau_gi'][0][0], time_step = args['time_step'], device = args['device'], spike_fn = args['spike_fn'])
        other_b = args['neuron_type'](inputs = b, weights = w1, v_rest_e = args['v_rest_e'][0][0], v_reset_e = args['v_reset_e'][0][0], v_thresh_e = args['v_thresh_e'][0][0], refrac_e = args['refrac_e'][0][0], tau_v = args['tau_v'][0][0], del_theta = args['del_theta'][0][0], ge_max = args['ge_max'][0][0], gi_max = args['gi_max'][0][0], tau_ge = args['tau_ge'][0][0], tau_gi = args['tau_gi'][0][0], time_step = args['time_step'], device = args['device'], spike_fn = args['spike_fn'])
        other_c = args['neuron_type'](inputs = c, weights = w1, v_rest_e = args['v_rest_e'][0][0], v_reset_e = args['v_reset_e'][0][0], v_thresh_e = args['v_thresh_e'][0][0], refrac_e = args['refrac_e'][0][0], tau_v = args['tau_v'][0][0], del_theta = args['del_theta'][0][0], ge_max = args['ge_max'][0][0], gi_max = args['gi_max'][0][0], tau_ge = args['tau_ge'][0][0], tau_gi = args['tau_gi'][0][0], time_step = args['time_step'], device = args['device'], spike_fn = args['spike_fn'])
    else:
        other_a = args['neuron_type'](inputs = a, weights = w1, tau_syn=args['tau_syn'][0][0], tau_mem=args['tau_mem'][0][0], device=args['device'], spike_fn=args['spike_fn'], time_step=args['time_step'])
        other_b = args['neuron_type'](inputs = b, weights = w1, tau_syn=args['tau_syn'][0][0], tau_mem=args['tau_mem'][0][0], device=args['device'], spike_fn=args['spike_fn'], time_step=args['time_step'])
        other_c = args['neuron_type'](inputs = c, weights = w1, tau_syn=args['tau_syn'][0][0], tau_mem=args['tau_mem'][0][0], device=args['device'], spike_fn=args['spike_fn'], time_step=args['time_step'])

    fig, axes = plt.subplots(nrows=3, ncols=3)
    axes[0,0].plot(c[0,:,0].tolist()  )
    axes[1,0].plot(other_c[0][0,:,0].tolist())
    axes[2,0].plot(other_c[1][0,:,0].tolist())


    axes[0,1].plot(a[0,:,0].tolist()  )
    axes[1,1].plot(other_a[0][0,:,0].tolist())
    axes[2,1].plot(other_a[1][0,:,0].tolist())


    axes[0,2].plot(b[0,:,0].tolist()  )
    axes[1,2].plot(other_b[0][0,:,0].tolist())
    axes[2,2].plot(other_b[1][0,:,0].tolist())

    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig.suptitle(args['neuron_type'].__name__)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figures/neuron_test_'+args['neuron_type'].__name__+"_"+str('{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() ))+".png",  dpi=300)


def weight_visual(weights, input_tuple):
    data = weights
    min_w = torch.min(data)
    data = data - min_w
    max_w = torch.max(data)
    data = data * (1/max_w) 

    images_len = weights.shape[1]

    plt.clf()
    fig, axes = plt.subplots(nrows=images_len, ncols=1, figsize=(3,30))
    for i in range(images_len):
        test = data[:,i]
        test = test.reshape(input_tuple)
        axes[i].imshow(test.detach().cpu().numpy(), cmap='gray')
    fig.tight_layout()
    plt.savefig("blabla.png", dpi=100)
