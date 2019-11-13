import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import torch


crossbar_color = "r"
pb_bmp_color = "k"
pb_csr_color = "b"
func_rowcol_color = "tab:orange"
func_bmp_color = "m"
func_pure_color = "g"


def step_d(bits): 
    return 2.0 ** (bits - 1)

def shift(x):
    if x == 0:
        return 1
    return 2 ** torch.round(torch.log(x) / np.log(2))

def clip(x, bits):
    if bits == 1:
        delta = 0.
    else:
        delta = 1./step_d(bits)
    maxv = +1 - delta
    minv = -1 + delta
    return torch.clamp(x, float(minv), float(maxv))

def quant(x, bits):
    if bits == 1: # BNN
        return torch.sign(x)
    else:
        scale = step_d(bits)
        return torch.round(x * scale ) / scale

def quant_w(x, global_wb, scale):
    with torch.no_grad():
        y = quant(clip(x, global_wb) , global_wb)
        diff = (y - x)

    #if scale <= 1.8:
    #    return x + diff
    return (x + diff)/scale


global_wb_list = [2,3,4,5]
global_ab_list = [6,7,8,10,12]
global_gb_list = [8,9,10,11,12]


scale1 = {2:16, 3:16, 4:16, 5:16, 6:16}
scale2 = {2:8, 3:8, 4:8, 5:8, 6:16}

data = pd.read_csv('./results/clee_out.csv')

x_list = []
y_list = []
y_real_ds1 = []
y_real_ds2 = []

# #### show configs needed
for w_c in global_wb_list:
	for a_c in global_ab_list:
		try:
			with open("./results/torch_wage_acc_cifar10_"+str(w_c)+str(a_c)+"8"+str(a_c)+".pkl", 'rb') as f:
				results = pickle.load(f)
			print(results['bit_string'])
			print(float(np.max(results['test_acc'])))
			
			print(results['test_acc'].index(np.max(results['test_acc'])))
			y_list.append(float(results['test_acc'].index(np.max(results['test_acc']))))

			tue_clee = data[data['Weight Bits'] == w_c]
			tue_clee = tue_clee[tue_clee['Activation Bits'] == a_c]
			tue_clee = tue_clee[tue_clee['Gradient Bits'] == 8]
			tue_clee = tue_clee[tue_clee['Dataset'] == 'cifar']

			tue_clee_ds1 = tue_clee[tue_clee['Data Structure'] == "PB-BMP"]
			tue_clee_ds2 = tue_clee[tue_clee['Data Structure'] == "PB-CSR"]




			temp1 = float(tue_clee_ds2['Energy']) * y_list[-1]
			temp2 = float(tue_clee_ds1['Energy']) * y_list[-1]
			temp3 = np.max(results['test_acc'])

			y_real_ds2.append(temp1)
			y_real_ds1.append(temp2)
			x_list.append(temp3)

			#qw1 = quant_w(results['w1'], w_c, scale1[w_c])
			#qw2 = quant_w(results['w2'], w_c, scale1[w_c])
			#x1 = torch.sum(qw1 != 0).float()/(qw1.shape[0] * qw1.shape[1])
			#x2 = torch.sum(qw2 != 0).float()/(qw2.shape[0] * qw2.shape[1])
			#print("wb, "+str(w_c)+", ab, "+str(a_c)+", eb, "+str(a_c)+", gb, 8"+", density, "+str(np.round(float(x1),2))+", image, 700")
			#print("wb, "+str(w_c)+", ab, "+str(a_c)+", eb, "+str(a_c)+", gb, 8"+", density, "+str(np.round(float(x2),2))+", image, 400")
		except:
			pass

for w_c in global_wb_list:
	for g_c in global_gb_list:
		try:
			with open("./results/torch_wage_acc_cifar10_"+str(w_c)+str(8)+str(g_c)+str(8)+".pkl", 'rb') as f:
				results = pickle.load(f)
			print(results['bit_string'])
			print(float(np.max(results['test_acc'])))
			
			print(results['test_acc'].index(np.max(results['test_acc'])))
			y_list.append(float(results['test_acc'].index(np.max(results['test_acc']))))

			tue_clee = data[data['Weight Bits'] == w_c]
			tue_clee = tue_clee[tue_clee['Activation Bits'] == 8]
			tue_clee = tue_clee[tue_clee['Gradient Bits'] == g_c]
			tue_clee = tue_clee[tue_clee['Dataset'] == 'cifar']

			tue_clee_ds1 = tue_clee[tue_clee['Data Structure'] == "PB-BMP"]
			tue_clee_ds2 = tue_clee[tue_clee['Data Structure'] == "PB-CSR"]


			temp1 = float(tue_clee_ds2['Energy']) * y_list[-1]
			temp2 = float(tue_clee_ds1['Energy']) * y_list[-1]
			temp3 = np.max(results['test_acc'])

			y_real_ds2.append(temp1)
			y_real_ds1.append(temp2)
			x_list.append(temp3)

			#qw1 = quant_w(results['w1'], w_c, scale1[w_c])
			#qw2 = quant_w(results['w2'], w_c, scale1[w_c])
			#x1 = torch.sum(qw1 != 0).float()/(qw1.shape[0] * qw1.shape[1])
			#x2 = torch.sum(qw2 != 0).float()/(qw2.shape[0] * qw2.shape[1])
			#print("wb, "+str(w_c)+", ab, "+str(12)+", eb, "+str(12)+", gb, "+str(g_c)+", density, "+str(np.round(float(x1),2))+", image, 700")
			#print("wb, "+str(w_c)+", ab, "+str(a_c)+", eb, "+str(a_c)+", gb, 8"+", density, "+str(np.round(float(x2),2))+", image, 400")
		except:
			pass



plt.clf()
plt.rc('font', weight='bold')
plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=1, ncols=1)

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)
#axes.set_xscale('log')

axes.scatter(y_real_ds1, x_list, marker="x", color="k", linewidth=2.5, label="PB-BMP")
axes.scatter(y_real_ds2, x_list, marker="*", color="b", linewidth=2.5, label="PB-CSR")
axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False)
axes.set_xlabel("Energy", fontweight='bold') #, fontsize=14, fontweight='bold'
axes.set_ylabel("Test Acc", fontweight='bold')

plt.tight_layout()
plt.savefig('./figures/tuesday_cifar.png')











x_list = []
y_list = []
y_real_ds1 = []
y_real_ds2 = []

# #### show configs needed
for w_c in global_wb_list:
	for a_c in global_ab_list:
		try:
			with open("./results/torch_wage_acc_mnist_"+str(w_c)+str(a_c)+"8"+str(a_c)+".pkl", 'rb') as f:
				results = pickle.load(f)
			print(results['bit_string'])
			print(float(np.max(results['test_acc'])))
			
			print(results['test_acc'].index(np.max(results['test_acc'])))
			y_list.append(float(results['test_acc'].index(np.max(results['test_acc']))))

			tue_clee = data[data['Weight Bits'] == w_c]
			tue_clee = tue_clee[tue_clee['Activation Bits'] == a_c]
			tue_clee = tue_clee[tue_clee['Gradient Bits'] == 8]

			tue_clee_ds1 = tue_clee[tue_clee['Data Structure'] == "PB-BMP"]
			tue_clee_ds2 = tue_clee[tue_clee['Data Structure'] == "PB-CSR"]
			tue_clee = tue_clee[tue_clee['Dataset'] == 'mnist']


			temp1 = float(tue_clee_ds2['Energy']) * y_list[-1]
			temp2 = float(tue_clee_ds1['Energy']) * y_list[-1]
			temp3 = np.max(results['test_acc'])

			y_real_ds2.append(temp1)
			y_real_ds1.append(temp2)
			x_list.append(temp3)

			#qw1 = quant_w(results['w1'], w_c, scale1[w_c])
			#qw2 = quant_w(results['w2'], w_c, scale1[w_c])
			#x1 = torch.sum(qw1 != 0).float()/(qw1.shape[0] * qw1.shape[1])
			#x2 = torch.sum(qw2 != 0).float()/(qw2.shape[0] * qw2.shape[1])
			#print("wb, "+str(w_c)+", ab, "+str(a_c)+", eb, "+str(a_c)+", gb, 8"+", density, "+str(np.round(float(x1),2))+", image, 700")
			#print("wb, "+str(w_c)+", ab, "+str(a_c)+", eb, "+str(a_c)+", gb, 8"+", density, "+str(np.round(float(x2),2))+", image, 400")
		except:
			pass

for w_c in global_wb_list:
	for g_c in global_gb_list:
		try:
			with open("./results/torch_wage_acc_mnist_"+str(w_c)+str(8)+str(g_c)+str(8)+".pkl", 'rb') as f:
				results = pickle.load(f)
			print(results['bit_string'])
			print(float(np.max(results['test_acc'])))
			
			print(results['test_acc'].index(np.max(results['test_acc'])))
			y_list.append(float(results['test_acc'].index(np.max(results['test_acc']))))

			tue_clee = data[data['Weight Bits'] == w_c]
			tue_clee = tue_clee[tue_clee['Activation Bits'] == 8]
			tue_clee = tue_clee[tue_clee['Gradient Bits'] == g_c]
			tue_clee = tue_clee[tue_clee['Dataset'] == 'mnist']

			tue_clee_ds1 = tue_clee[tue_clee['Data Structure'] == "PB-BMP"]
			tue_clee_ds2 = tue_clee[tue_clee['Data Structure'] == "PB-CSR"]


			temp1 = float(tue_clee_ds2['Energy']) * y_list[-1]
			temp2 = float(tue_clee_ds1['Energy']) * y_list[-1]
			temp3 = np.max(results['test_acc'])

			y_real_ds2.append(temp1)
			y_real_ds1.append(temp2)
			x_list.append(temp3)

			#qw1 = quant_w(results['w1'], w_c, scale1[w_c])
			#qw2 = quant_w(results['w2'], w_c, scale1[w_c])
			#x1 = torch.sum(qw1 != 0).float()/(qw1.shape[0] * qw1.shape[1])
			#x2 = torch.sum(qw2 != 0).float()/(qw2.shape[0] * qw2.shape[1])
			#print("wb, "+str(w_c)+", ab, "+str(12)+", eb, "+str(12)+", gb, "+str(g_c)+", density, "+str(np.round(float(x1),2))+", image, 700")
			#print("wb, "+str(w_c)+", ab, "+str(a_c)+", eb, "+str(a_c)+", gb, 8"+", density, "+str(np.round(float(x2),2))+", image, 400")
		except:
			pass



plt.clf()
plt.rc('font', weight='bold')
plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=1, ncols=1)

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)
#axes.set_xscale('log')

axes.scatter(y_real_ds1, x_list, marker="x", color="k", linewidth=2.5, label="PB-BMP")
axes.scatter(y_real_ds2, x_list, marker="*", color="b", linewidth=2.5, label="PB-CSR")
axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False)
axes.set_xlabel("Energy", fontweight='bold') #, fontsize=14, fontweight='bold'
axes.set_ylabel("Test Acc", fontweight='bold')

plt.tight_layout()
plt.savefig('./figures/tuesday_mnist.png')

# data = pd.read_csv('/Users/clemens/Documents/Spiking Neural Networks/QuantizedSNN/iscas_figs/frontier.csv')
# test = data[data['Layer'] == "FullyConnected"]
# test = test[test['Channels'] == 1]
# test = test[test['Filter Size'] == 1]
# #test = test[test['Image Size'] == 28]

# #imp_list = []


# # need to use last loss!
# fin_ds1_y = []
# fin_ds1_x = []
# fin_ds2_x = []
# fin_ds3_x = []


# for i in global_wb_list:
# 	# potenially another for loop for act/gradient + whole repeat

# 	ds1_y = []
# 	ds1_x = []
# 	ds2_x = []
# 	ds3_x = []

# 	# only best!
# 	for j in global_gb_list:
# 		with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+str(i)+"12"+str(j)+"12_1.0.pkl", 'rb') as f:
# 			results = pickle.load(f)

# 		qw1 = quant_w(results['w1'], i, scale1[i])
# 		qw2 = quant_w(results['w2'], i, scale1[i])
# 		x1 = torch.sum(qw1 == 0).float()/(qw1.shape[0] * qw1.shape[1])
# 		x2 = torch.sum(qw2 == 0).float()/(qw2.shape[0] * qw2.shape[1])
# 		ds1_y.append(np.min(results['loss_hist']))



# 		test_t = test[test['Weight Bits'] == i] #the weight bit level right now
# 		# potentially another line for the gradients quant level
# 		# and another line for act quant level
# 		spar_1 = test_t[test_t['Out Length'] == 400]
# 		spar_1 = spar_1[spar_1['Sparsity'] == np.round(1-float(x1),2)] #this one should look up computed sparsity
# 		spar_2 = test_t[test_t['Out Length'] == 250]
# 		spar_2 = spar_2[spar_2['Sparsity'] == np.round(1-float(x2),2)]


# 		spar_1 = spar_1.drop_duplicates()
# 		spar_2 = spar_2.drop_duplicates()

# 		#import pdb; pdb.set_trace()

# 		ds1 = spar_1[spar_1['Data Structure'] == 'Crossbar']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_1 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1 = spar_2[spar_2['Data Structure'] == 'Crossbar']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_2 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1_x.append(ds1_1.item() + ds1_2.item())



# 		ds1 = spar_1[spar_1['Data Structure'] == 'PB-BMP']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_1 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1 = spar_2[spar_2['Data Structure'] == 'PB-BMP']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_2 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds2_x.append(ds1_1.item() + ds1_2.item())


# 		ds1 = spar_1[spar_1['Data Structure'] == 'PB-CSR']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_1 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1 = spar_2[spar_2['Data Structure'] == 'PB-CSR']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_2 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds3_x.append(ds1_1.item() + ds1_2.item())

# 	# only best!
# 	for j in global_ab_list:
# 		with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+str(i)+str(j)+"8"+str(j)+"_1.0.pkl", 'rb') as f:
# 			results = pickle.load(f)


# 		qw1 = quant_w(results['w1'], i, scale1[i])
# 		qw2 = quant_w(results['w2'], i, scale1[i])
# 		x1 = torch.sum(qw1 == 0).float()/(qw1.shape[0] * qw1.shape[1])
# 		x2 = torch.sum(qw2 == 0).float()/(qw2.shape[0] * qw2.shape[1])
# 		ds1_y.append(np.min(results['loss_hist']))



# 		test_t = test[test['Weight Bits'] == i] #the weight bit level right now
# 		# potentially another line for the gradients quant level
# 		# and another line for act quant level
# 		spar_1 = test_t[test_t['Out Length'] == 400]
# 		spar_1 = spar_1[spar_1['Sparsity'] == np.round(1-float(x1),2)] #this one should look up computed sparsity
# 		spar_2 = test_t[test_t['Out Length'] == 250]
# 		spar_2 = spar_2[spar_2['Sparsity'] == np.round(1-float(x2),2)]


# 		spar_1 = spar_1.drop_duplicates()
# 		spar_2 = spar_2.drop_duplicates()

# 		#import pdb; pdb.set_trace()

# 		ds1 = spar_1[spar_1['Data Structure'] == 'Crossbar']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_1 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1 = spar_2[spar_2['Data Structure'] == 'Crossbar']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_2 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1_x.append(ds1_1.item() + ds1_2.item())



# 		ds1 = spar_1[spar_1['Data Structure'] == 'PB-BMP']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_1 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1 = spar_2[spar_2['Data Structure'] == 'PB-BMP']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_2 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds2_x.append(ds1_1.item() + ds1_2.item())


# 		ds1 = spar_1[spar_1['Data Structure'] == 'PB-CSR']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_1 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1 = spar_2[spar_2['Data Structure'] == 'PB-CSR']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_2 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds3_x.append(ds1_1.item() + ds1_2.item())


# 	#import pdb; pdb.set_trace()
# 	idx_best = np.argmin(ds1_y)
# 	fin_ds1_y.append(ds1_y[idx_best])
# 	fin_ds1_x.append(ds1_x[idx_best])
# 	fin_ds2_x.append(ds2_x[idx_best])
# 	fin_ds3_x.append(ds3_x[idx_best])



# plt.clf()
# plt.rc('font', weight='bold')
# plt.rc('font', size='14')
# fig, axes = plt.subplots(nrows=1, ncols=1)

# for axis in ['bottom','left']:
#   axes.spines[axis].set_linewidth(3)
# for axis in ['top','right']:
#   axes.spines[axis].set_linewidth(0)
# axes.xaxis.set_tick_params(width=2)
# axes.yaxis.set_tick_params(width=2)
# axes.set_xscale('log')

# axes.scatter(fin_ds1_x, fin_ds1_y,  linewidth=2.5, label="Crossbar", marker = "x", color=crossbar_color)
# axes.scatter(fin_ds2_x, fin_ds1_y,  linewidth=2.5, label="PB-BMP", marker = "s", color=pb_bmp_color)
# axes.scatter(fin_ds3_x, fin_ds1_y, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
# axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False)
# axes.set_xlabel("Energy (J)", fontweight='bold') #, fontsize=14, fontweight='bold'
# axes.set_ylabel("Van Rossum Distance", fontweight='bold')

# plt.tight_layout()
# plt.savefig('/Users/clemens/Desktop/cloud.png')


#2.758796843901395e-05  # pb-bmp
#2.305499313403695e-05  # crossbar


# 2.0306438524536953e-05 #crossbar
# 1.683712824950995e-05

# for i in global_wb_list:
# 	# potenially another for loop for act/gradient + whole repeat

# 	ds1_y = []
# 	ds1_x = []
# 	ds2_x = []
# 	ds3_x = []

# 	# only best!
# 	for j in global_ab_list:
# 		with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+str(i)+str(j)+"8"+str(j)+"_1.0.pkl", 'rb') as f:
# 			results = pickle.load(f)


# 		qw1 = quant_w(results['w1'], i, scale1[i])
# 		qw2 = quant_w(results['w2'], i, scale1[i])
# 		x1 = torch.sum(qw1 == 0).float()/(qw1.shape[0] * qw1.shape[1])
# 		x2 = torch.sum(qw2 == 0).float()/(qw2.shape[0] * qw2.shape[1])
# 		ds1_y.append(np.min(results['loss_hist']))



# 		test_t = test[test['Weight Bits'] == i] #the weight bit level right now
# 		# potentially another line for the gradients quant level
# 		# and another line for act quant level
# 		spar_1 = test_t[test_t['Out Length'] == 400]
# 		spar_1 = spar_1[spar_1['Sparsity'] == np.round(1-float(x1),2)] #this one should look up computed sparsity
# 		spar_2 = test_t[test_t['Out Length'] == 250]
# 		spar_2 = spar_2[spar_2['Sparsity'] == np.round(1-float(x2),2)]


# 		spar_1 = spar_1.drop_duplicates()
# 		spar_2 = spar_2.drop_duplicates()

# 		#import pdb; pdb.set_trace()

# 		ds1 = spar_1[spar_1['Data Structure'] == 'Crossbar']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_1 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1 = spar_2[spar_2['Data Structure'] == 'Crossbar']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_2 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1_x.append(ds1_1.item() + ds1_2.item())



# 		ds1 = spar_1[spar_1['Data Structure'] == 'PB-BMP']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_1 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1 = spar_2[spar_2['Data Structure'] == 'PB-BMP']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_2 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds2_x.append(ds1_1.item() + ds1_2.item())


# 		ds1 = spar_1[spar_1['Data Structure'] == 'PB-CSR']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_1 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds1 = spar_2[spar_2['Data Structure'] == 'PB-CSR']
# 		ds1 = ds1.drop_duplicates()
# 		ds1_2 = ds1['Forward Energy'] + ds1['Backward Energy']
# 		ds3_x.append(ds1_1.item() + ds1_2.item())


