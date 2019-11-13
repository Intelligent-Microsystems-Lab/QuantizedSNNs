import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd



with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_34343434_1.0.pkl", 'rb') as f:
	results = pickle.load(f)


### 2888
with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_2888_1.0.pkl", 'rb') as f:
	results = pickle.load(f)

####
data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')
test = data[data['Layer'] == "FullyConnected"]
test = test[test['Channels'] == 32]
test = test[test['Sparsity'] == 1]
test = test[test['Filter Size'] == 3]
test = test[test['Image Size'] == 28]
test = test[test['Weight Bits'] == 2]

ds1 = test[test['Data Structure'] == 'Crossbar']
mult_val1 = ds1['Forward Energy'] + ds1['Backward Energy'] + ds1['Update Energy']
energy_level1 = np.arange(1,len(results['loss_hist'])+1)*float(mult_val1)

ds2 = test[test['Data Structure'] == 'PB-BMP']
mult_val2 = ds2['Forward Energy'] + ds2['Backward Energy'] + ds2['Update Energy']
energy_level2 = np.arange(1,len(results['loss_hist'])+1)*float(mult_val2)

ds3 = test[test['Data Structure'] == 'PB-CSR']
mult_val3 = ds3['Forward Energy'] + ds3['Backward Energy'] + ds3['Update Energy']
energy_level3 = np.arange(1,len(results['loss_hist'])+1)*float(mult_val3)



plt.clf()
plt.rc('font', weight='bold')
fig, axes = plt.subplots(nrows=1, ncols=1)

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)

axes.plot(energy_level1, results['loss_hist'],  linewidth=2.5, label="Crossbar", color="tab:blue")
axes.plot(energy_level2, results['loss_hist'],  linewidth=2.5, label="PB-BMP", color="tab:orange")
axes.plot(energy_level3, results['loss_hist'], linewidth=2.5, label="PB-CSR", color="tab:green")
axes.legend(frameon=False)
axes.set_xlabel("Energy (J)", fontweight='bold') #, fontsize=14, fontweight='bold'
axes.set_ylabel("Van Rossum Distance", fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/clemens/Desktop/2888.png')





### 6888
with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_6888_1.0.pkl", 'rb') as f:
	results = pickle.load(f)

####
data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')
test = data[data['Layer'] == "FullyConnected"]
test = test[test['Channels'] == 32]
test = test[test['Sparsity'] == 1]
test = test[test['Filter Size'] == 3]
test = test[test['Image Size'] == 28]
test = test[test['Weight Bits'] == 6]

ds1 = test[test['Data Structure'] == 'Crossbar']
mult_val1 = ds1['Forward Energy'] + ds1['Backward Energy'] + ds1['Update Energy']
energy_level1 = np.arange(1,len(results['loss_hist'])+1)*float(mult_val1)

ds2 = test[test['Data Structure'] == 'PB-BMP']
mult_val2 = ds2['Forward Energy'] + ds2['Backward Energy'] + ds2['Update Energy']
energy_level2 = np.arange(1,len(results['loss_hist'])+1)*float(mult_val2)

ds3 = test[test['Data Structure'] == 'PB-CSR']
mult_val3 = ds3['Forward Energy'] + ds3['Backward Energy'] + ds3['Update Energy']
energy_level3 = np.arange(1,len(results['loss_hist'])+1)*float(mult_val3)



plt.clf()
plt.rc('font', weight='bold')
fig, axes = plt.subplots(nrows=1, ncols=1)

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)

axes.plot(energy_level1, results['loss_hist'],  linewidth=2.5, label="Crossbar", color="tab:blue")
axes.plot(energy_level2, results['loss_hist'],  linewidth=2.5, label="PB-BMP", color="tab:orange")
axes.plot(energy_level3, results['loss_hist'], linewidth=2.5, label="PB-CSR", color="tab:green")
axes.legend(frameon=False)
axes.set_xlabel("Energy (J)", fontweight='bold') #, fontsize=14, fontweight='bold'
axes.set_ylabel("Van Rossum Distance", fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/clemens/Desktop/6888.png')

