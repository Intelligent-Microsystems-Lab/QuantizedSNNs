import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd


# results graph
data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')


# conv
test = data[data['Layer'] == "FullyConnected"]
test = test[test['Channels'] == 32]
test = test[test['Sparsity'] == 1]
test = test[test['Filter Size'] == 3]
test = test[test['Image Size'] == 28]

ds1 = test[test['Data Structure'] == 'Crossbar']
ds1 = ds1.drop_duplicates()
ds1_forward = ds1['Forward Energy']
ds1_backward = ds1['Backward Energy'] 
ds1_update =  ds1['Update Energy']


ds2 = test[test['Data Structure'] == 'PB-BMP']
ds2 = ds2.drop_duplicates()
ds2_forward = ds2['Forward Energy']
ds2_backward = ds2['Backward Energy'] 
ds2_update =  ds2['Update Energy']


ds3 = test[test['Data Structure'] == 'PB-CSR']
ds3 = ds3.drop_duplicates()
ds3_forward = ds3['Forward Energy']
ds3_backward = ds3['Backward Energy']
ds3_update= ds3['Update Energy']


plt.clf()

#plt.rc('font', family='serif')
plt.rc('font', weight='bold')
fig, axes = plt.subplots(nrows=3, ncols=1)

#plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'

for axis in ['bottom','left']:
  axes[0].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[0].spines[axis].set_linewidth(0)
axes[0].xaxis.set_tick_params(width=2)
axes[0].yaxis.set_tick_params(width=2)


for axis in ['bottom','left']:
  axes[1].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[1].spines[axis].set_linewidth(0)
axes[1].xaxis.set_tick_params(width=2)
axes[1].yaxis.set_tick_params(width=2)

for axis in ['bottom','left']:
  axes[2].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[2].spines[axis].set_linewidth(0)
axes[2].xaxis.set_tick_params(width=2)
axes[2].yaxis.set_tick_params(width=2)


#axes[0].set_title("Forward Pass")
axes[0].plot(np.arange(2,13,1), ds1_forward,  linewidth=2.5, label="Crossbar", color="tab:blue")
axes[0].plot(np.arange(2,13,1), ds2_forward,  linewidth=2.5, label="PB-BMP", color="tab:orange")
axes[0].plot(np.arange(2,13,1), ds3_forward, linewidth=2.5, label="PB-CSR", color="tab:green")
axes[0].legend(frameon=False)
axes[0].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[0].set_ylabel("Energy (J)", fontweight='bold')


#axes[1].set_title("Backward Pass")
axes[1].plot(np.arange(2,13,1), ds1_backward, linewidth=2.5, label="Crossbar", color="tab:blue")
axes[1].plot(np.arange(2,13,1), ds2_backward, linewidth=2.5, label="PB-BMP", color="tab:orange")
axes[1].plot(np.arange(2,13,1),ds3_backward, linewidth=2.5, label="PB-CSR", color="tab:green")
axes[1].legend(frameon=False)
axes[1].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[1].set_ylabel("Energy (J)", fontweight='bold')


#axes[1].set_title("Backward Pass")
axes[2].plot(np.arange(2,13,1), ds1_update, linewidth=2.5, label="Crossbar", color="tab:blue")
axes[2].plot(np.arange(2,13,1), ds2_update, linewidth=2.5, label="PB-BMP", color="tab:orange")
axes[2].plot(np.arange(2,13,1), ds3_update, linewidth=2.5, label="PB-CSR", color="tab:green")
axes[2].legend(frameon=False)
axes[2].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[2].set_ylabel("Energy (J)", fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/clemens/Desktop/fc_up.png')
#plt.show()

#####
##### Convolutional
#####

# results graph
data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')


# conv
test = data[data['Layer'] == "Convolution"]
test = test[test['Channels'] == 32]
test = test[test['Sparsity'] == 1]
test = test[test['Filter Size'] == 3]
test = test[test['Image Size'] == 28]

ds1 = test[test['Data Structure'] == 'Func-RowCol']
ds1 = ds1.drop_duplicates()
ds1_forward = ds1['Forward Energy']
ds1_backward = ds1['Backward Energy'] 
ds1_update = ds1['Update Energy']


ds2 = test[test['Data Structure'] == 'PB-BMP']
ds2 = ds2.drop_duplicates()
ds2_forward = ds2['Forward Energy']
ds2_backward = ds2['Backward Energy']
ds2_update =  ds2['Update Energy']


ds3 = test[test['Data Structure'] == 'PB-CSR']
ds3 = ds3.drop_duplicates()
ds3_forward = ds3['Forward Energy']
ds3_backward = ds3['Backward Energy'] 
ds3_update =   ds3['Update Energy']


ds4 = test[test['Data Structure'] == 'Func-BMP']
ds4 = ds4.drop_duplicates()
ds4_forward = ds4['Forward Energy']
ds4_backward = ds4['Backward Energy'] 
ds4_update =  ds4['Update Energy']

ds5 = test[test['Data Structure'] == 'Func-Pure']
ds5 = ds5.drop_duplicates()
ds5_forward = ds5['Forward Energy']
ds5_backward = ds5['Backward Energy']
ds5_update =  ds5['Update Energy']

plt.clf()

#plt.rc('font', family='serif')
plt.rc('font', weight='bold')
fig, axes = plt.subplots(nrows=3, ncols=1)

#plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'

for axis in ['bottom','left']:
  axes[0].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[0].spines[axis].set_linewidth(0)
axes[0].xaxis.set_tick_params(width=2)
axes[0].yaxis.set_tick_params(width=2)


for axis in ['bottom','left']:
  axes[1].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[1].spines[axis].set_linewidth(0)
axes[1].xaxis.set_tick_params(width=2)
axes[1].yaxis.set_tick_params(width=2)


for axis in ['bottom','left']:
  axes[2].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[2].spines[axis].set_linewidth(0)
axes[2].xaxis.set_tick_params(width=2)
axes[2].yaxis.set_tick_params(width=2)


#axes[0].set_title("Forward Pass")
axes[0].plot(np.arange(2,13,1), ds1_forward,  linewidth=2.5, label="Func-RowCol", color="tab:red")
axes[0].plot(np.arange(2,13,1), ds2_forward,  linewidth=2.5, label="PB-BMP", color="tab:orange")
axes[0].plot(np.arange(2,13,1), ds3_forward, linewidth=2.5, label="PB-CSR", color="tab:green")
axes[0].plot(np.arange(2,13,1), ds4_forward,  linewidth=2.5, label="Func-BMP", color="tab:purple")
axes[0].plot(np.arange(2,13,1), ds5_forward,  linewidth=2.5, label="Func-Pure", color="tab:brown")
axes[0].legend(frameon=False)
axes[0].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[0].set_ylabel("Energy (J)", fontweight='bold')


#axes[1].set_title("Backward Pass")
axes[1].plot(np.arange(2,13,1), ds1_backward, linewidth=2.5, label="Func-RowCol", color="tab:red")
axes[1].plot(np.arange(2,13,1), ds2_backward, linewidth=2.5, label="PB-BMP", color="tab:orange")
axes[1].plot(np.arange(2,13,1), ds3_backward, linewidth=2.5, label="PB-CSR", color="tab:green")
axes[1].plot(np.arange(2,13,1), ds4_backward, linewidth=2.5, label="Func-BMP", color="tab:purple")
axes[1].plot(np.arange(2,13,1), ds5_backward, linewidth=2.5, label="Func-Pure", color="tab:brown")
axes[1].legend(frameon=False)
axes[1].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[1].set_ylabel("Energy (J)", fontweight='bold')

#axes[1].set_title("Backward Pass")
axes[2].plot(np.arange(2,13,1), ds1_update, linewidth=2.5, label="Func-RowCol", color="tab:red")
axes[2].plot(np.arange(2,13,1), ds2_update, linewidth=2.5, label="PB-BMP", color="tab:orange")
axes[2].plot(np.arange(2,13,1), ds3_update, linewidth=2.5, label="PB-CSR", color="tab:green")
axes[2].plot(np.arange(2,13,1), ds4_update, linewidth=2.5, label="Func-BMP", color="tab:purple")
axes[2].plot(np.arange(2,13,1), ds5_update, linewidth=2.5, label="Func-Pure", color="tab:brown")
axes[2].legend(frameon=False)
axes[2].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[2].set_ylabel("Energy (J)", fontweight='bold')



plt.tight_layout()
plt.savefig('/Users/clemens/Desktop/conv_up.png')
#plt.show()


####
#### sparsity
####
data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')

# conv
test = data[data['Layer'] == "FullyConnected"]
test = test[test['Channels'] == 64]
test = test[test['Filter Size'] == 3]
test = test[test['Image Size'] == 28]
test = test[test['Weight Bits'] == 8]
test = test.drop_duplicates()
test = test.sort_values(by=['Sparsity'])

ds1 = test[test['Data Structure'] == 'Crossbar']
ds1 = ds1.drop_duplicates()
ds1_forward = ds1['Forward Energy']
ds1_backward = ds1['Backward Energy']
ds1_update =  ds1['Update Energy']


ds2 = test[test['Data Structure'] == 'PB-BMP']
ds2 = ds2.drop_duplicates()
ds2_forward = ds2['Forward Energy']
ds2_backward = ds2['Backward Energy']
ds2_update = ds2['Update Energy']


ds3 = test[test['Data Structure'] == 'PB-CSR']
ds3 = ds3.drop_duplicates()
ds3_forward = ds3['Forward Energy']
ds3_backward = ds3['Backward Energy']
ds3_update = ds3['Update Energy']

plt.clf()



#plt.rc('font', family='serif')
plt.rc('font', weight='bold')
fig, axes = plt.subplots(nrows=3, ncols=1)

#plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'

for axis in ['bottom','left']:
  axes[0].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[0].spines[axis].set_linewidth(0)
axes[0].xaxis.set_tick_params(width=2)
axes[0].yaxis.set_tick_params(width=2)


for axis in ['bottom','left']:
  axes[1].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[1].spines[axis].set_linewidth(0)
axes[1].xaxis.set_tick_params(width=2)
axes[1].yaxis.set_tick_params(width=2)


for axis in ['bottom','left']:
  axes[2].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[2].spines[axis].set_linewidth(0)
axes[2].xaxis.set_tick_params(width=2)
axes[2].yaxis.set_tick_params(width=2)

#axes[0].set_title("Forward Pass")
axes[0].plot(np.arange(.05,1,.05), ds1_forward,  linewidth=2.5, label="Crossbar", color="tab:blue")
axes[0].plot(np.arange(.05,1,.05), ds2_forward,  linewidth=2.5, label="PB-BMP", color="tab:orange")
axes[0].plot(np.arange(.05,1,.05), ds3_forward, linewidth=2.5, label="PB-CSR", color="tab:green")
axes[0].legend(frameon=False)
axes[0].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[0].set_ylabel("Energy (J)", fontweight='bold')


#axes[1].set_title("Backward Pass")
axes[1].plot(np.arange(.05,1,.05), ds1_backward, linewidth=2.5, label="Crossbar", color="tab:blue")
axes[1].plot(np.arange(.05,1,.05), ds2_backward, linewidth=2.5, label="PB-BMP", color="tab:orange")
axes[1].plot(np.arange(.05,1,.05),ds3_backward, linewidth=2.5, label="PB-CSR", color="tab:green")
axes[1].legend(frameon=False)
axes[1].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[1].set_ylabel("Energy (J)", fontweight='bold')


axes[2].plot(np.arange(.05,1,.05), ds1_update, linewidth=2.5, label="Crossbar", color="tab:blue")
axes[2].plot(np.arange(.05,1,.05), ds2_update, linewidth=2.5, label="PB-BMP", color="tab:orange")
axes[2].plot(np.arange(.05,1,.05), ds3_update, linewidth=2.5, label="PB-CSR", color="tab:green")
axes[2].legend(frameon=False)
axes[2].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[2].set_ylabel("Energy (J)", fontweight='bold')


plt.tight_layout()
plt.savefig('/Users/clemens/Desktop/up_sparsity.png')



####
#### alpha sweep plot
####
data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')

# conv
test = data[data['Layer'] == "FullyConnected"]
test = test[test['Channels'] == 32]
test = test[test['Filter Size'] == 3]
test = test[test['Image Size'] == 28]
test = test[test['Sparsity'] == 1]
test = test[test['Weight Bits'] == 8]
test = test.drop_duplicates()


ds1 = test[test['Data Structure'] == 'Crossbar']
ds1 = ds1.drop_duplicates()
ds1_forward = ds1['Forward Energy']
ds1_backward = ds1['Backward Energy']
ds1_update =  ds1['Update Energy']


ds2 = test[test['Data Structure'] == 'PB-BMP']
ds2 = ds2.drop_duplicates()
ds2_forward = ds2['Forward Energy']
ds2_backward = ds2['Backward Energy']
ds2_update = ds2['Update Energy']


ds3 = test[test['Data Structure'] == 'PB-CSR']
ds3 = ds3.drop_duplicates()
ds3_forward = ds3['Forward Energy']
ds3_backward = ds3['Backward Energy']
ds3_update = ds3['Update Energy']





