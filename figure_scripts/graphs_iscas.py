import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import pandas as pd


plt.rc('font', weight='bold')
plt.rc('font', size='19')
#plt.rc('formatter',useoffset=True )
#mpl.rcParams['axes.formatter.useoffset'] = True

crossbar_color = "r"
pb_bmp_color = "k"
pb_csr_color = "b"
func_rowcol_color = "tab:orange"
func_bmp_color = "m"
func_pure_color = "g"


# results graph
data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')

# fc
test = data[data['Layer'] == "FullyConnected"]
test = test[test['Channels'] == 32]
test = test[test['Sparsity'] == .25]
test = test[test['Filter Size'] == 3]
test = test[test['Image Size'] == 28]

ds1 = test[test['Data Structure'] == 'Crossbar']
ds1 = ds1.drop_duplicates()
ds1_forward = ds1['Forward Energy']
ds1_backward = ds1['Backward Energy'] 


ds2 = test[test['Data Structure'] == 'PB-BMP']
ds2 = ds2.drop_duplicates()
ds2_forward = ds2['Forward Energy']
ds2_backward = ds2['Backward Energy'] 


ds3 = test[test['Data Structure'] == 'PB-CSR']
ds3 = ds3.drop_duplicates()
ds3_forward = ds3['Forward Energy']
ds3_backward = ds3['Backward Energy']


plt.clf()

#plt.rc('font', family='serif')
plt.rc('font', weight='bold')
#plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=2, ncols=1) #

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

#import pdb; pdb.set_trace()
#axes[0].set_title("Forward Pass")
axes[0].set_yscale('linear')
#axes[0].get_yaxis().get_major_formatter().set_useOffset(True)
#axes[0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
#axes[0].ticklabel_format(axis='y', style='sci')
axes[0].plot(np.arange(2,13,1), ds1_forward*10**6,  linewidth=2.5, label="Crossbar", color=crossbar_color)
axes[0].plot(np.arange(2,13,1), ds2_forward*10**6,  linewidth=2.5, label="PB-BMP", color=pb_bmp_color)
axes[0].plot(np.arange(2,13,1), ds3_forward*10**6, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
#axes[0].text(1, 2.3, 'x10e-5')

axes[0].legend(loc='lower center', bbox_to_anchor=(0.5,1.08), ncol=3, frameon=False)
axes[0].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[0].set_ylabel("Energy (mJ) ", fontweight='bold')
axes[0].ticklabel_format(style='sci', axis='y', useOffset=True)


#axes[1].set_title("Backward Pass")
axes[1].set_yscale('linear')
#axes[1].get_yaxis().get_major_formatter().set_useOffset(True)
#axes[1].ticklabel_format(axis='y', style='sci')
axes[1].plot(np.arange(2,13,1), ds1_backward*10**6, linewidth=2.5, label="Crossbar", color=crossbar_color)
axes[1].plot(np.arange(2,13,1), ds2_backward*10**6, linewidth=2.5, label="PB-BMP", color=pb_bmp_color)
axes[1].plot(np.arange(2,13,1), ds3_backward*10**6, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
#axes[1].text(1, 6, 'x10e-5')
#axes[1].legend(frameon=False)

axes[1].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[1].set_ylabel("Energy (mJ)", fontweight='bold')
axes[1].ticklabel_format(style='sci', axis='y', useOffset=True)
plt.ticklabel_format(style='sci', axis='y', useOffset=True)
plt.tight_layout()
plt.savefig('/Users/clemens/Desktop/fc.png')
#plt.show()


ds1_forward_old = ds1_forward
ds2_forward_old = ds2_forward
ds3_forward_old = ds3_forward

ds1_backward_old = ds1_backward
ds2_backward_old = ds2_backward
ds3_backward_old = ds3_backward

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


ds2 = test[test['Data Structure'] == 'PB-BMP']
ds2 = ds2.drop_duplicates()
ds2_forward = ds2['Forward Energy']
ds2_backward = ds2['Backward Energy'] 


ds3 = test[test['Data Structure'] == 'PB-CSR']
ds3 = ds3.drop_duplicates()
ds3_forward = ds3['Forward Energy']
ds3_backward = ds3['Backward Energy'] 


ds4 = test[test['Data Structure'] == 'Func-BMP']
ds4 = ds4.drop_duplicates()
ds4_forward = ds4['Forward Energy']
ds4_backward = ds4['Backward Energy'] 

ds5 = test[test['Data Structure'] == 'Func-Pure']
ds5 = ds5.drop_duplicates()
ds5_forward = ds5['Forward Energy']
ds5_backward = ds5['Backward Energy'] 


plt.clf()

#plt.rc('font', family='serif')
plt.rc('font', weight='bold')
#plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=2, ncols=1)

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


#axes[0].set_title("Forward Pass")
axes[0].set_yscale('linear')
axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
#axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
#axes[0].ticklabel_format(axis='y', style='sci')
axes[0].plot(np.arange(2,13,1), ds1_forward*10**6,  linewidth=2.5, label="Func-RowCol", color=func_rowcol_color)
axes[0].plot(np.arange(2,13,1), ds2_forward*10**6,  linewidth=2.5, label="PB-BMP", color=pb_bmp_color)
axes[0].plot(np.arange(2,13,1), ds3_forward*10**6, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
#axes[0].plot(np.arange(2,13,1), ds4_forward,  linewidth=2.5, label="Func-BMP", color=func_bmp_color)
#axes[0].plot(np.arange(2,13,1), ds5_forward,  linewidth=2.5, label="Func-Pure", color=func_pure_color)

axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
axes[0].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[0].set_ylabel("Energy (mJ)", fontweight='bold')


#axes[1].set_title("Backward Pass")
axes[1].set_yscale('linear')
#axes[1].get_yaxis().get_major_formatter().set_useOffset(False)
#axes[1].ticklabel_format(axis='y', style='sci')
axes[1].plot(np.arange(2,13,1), ds1_backward*10**6, linewidth=2.5, label="Func-RowCol", color=func_rowcol_color)
axes[1].plot(np.arange(2,13,1), ds2_backward*10**6, linewidth=2.5, label="PB-BMP", color=pb_bmp_color)
axes[1].plot(np.arange(2,13,1), ds3_backward*10**6, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
#axes[1].plot(np.arange(2,13,1), ds4_backward, linewidth=2.5, label="Func-BMP", color=func_bmp_color)
#axes[1].plot(np.arange(2,13,1), ds5_backward, linewidth=2.5, label="Func-Pure", color=func_pure_color)
#axes[1].legend(frameon=False)
axes[1].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[1].set_ylabel("Energy (mJ)", fontweight='bold')


plt.tight_layout()
plt.savefig('/Users/clemens/Desktop/conv.png')





############## Combined 

plt.clf()

#plt.rc('font', family='serif')
plt.rc('font', weight='bold')
#plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12.8,9.6))

#plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'



for axis in ['bottom','left']:
  axes[0,0].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[0,0].spines[axis].set_linewidth(0)
axes[0,0].xaxis.set_tick_params(width=2)
axes[0,0].yaxis.set_tick_params(width=2)


for axis in ['bottom','left']:
  axes[0,1].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[0,1].spines[axis].set_linewidth(0)
axes[0,1].xaxis.set_tick_params(width=2)
axes[0,1].yaxis.set_tick_params(width=2)

#import pdb; pdb.set_trace()
#axes[0].set_title("Forward Pass")
axes[0,0].set_yscale('linear')
#axes[0].get_yaxes().get_major_formatter().set_useOffset(True)
#axes[0].yaxes.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
#axes[0].ticklabel_format(axes='y', style='sci')
axes[0,0].plot(np.arange(2,13,1), ds1_forward_old*10**6,  linewidth=2.5, label="Crossbar", color=crossbar_color)
axes[0,0].plot(np.arange(2,13,1), ds2_forward_old*10**6,  linewidth=2.5, label="PB-BMP", color=pb_bmp_color)
axes[0,0].plot(np.arange(2,13,1), ds3_forward_old*10**6, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
#axes[0].text(1, 2.3, 'x10e-5')

axes[0,0].legend(loc='lower center', bbox_to_anchor=(0.5,1.08), ncol=3, frameon=False)
axes[0,0].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[0,0].set_ylabel("Energy (mJ) ", fontweight='bold')
axes[0,0].ticklabel_format(style='sci', axis='y', useOffset=True)


#axes[1].set_title("Backward Pass")
axes[0,1].set_yscale('linear')
#axes[1].get_yaxis().get_major_formatter().set_useOffset(True)
#axes[1].ticklabel_format(axis='y', style='sci')
axes[0,1].plot(np.arange(2,13,1), ds1_backward_old*10**6, linewidth=2.5, label="Crossbar", color=crossbar_color)
axes[0,1].plot(np.arange(2,13,1), ds2_backward_old*10**6, linewidth=2.5, label="PB-BMP", color=pb_bmp_color)
axes[0,1].plot(np.arange(2,13,1), ds3_backward_old*10**6, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
#axes[1].text(1, 6, 'x10e-5')
#axes[1].legend(frameon=False)

axes[0,1].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[0,1].set_ylabel("Energy (mJ)", fontweight='bold')
axes[0,1].ticklabel_format(style='sci', axis='y', useOffset=True)
plt.ticklabel_format(style='sci', axis='y', useOffset=True)





for axis in ['bottom','left']:
  axes[1,0].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[1,0].spines[axis].set_linewidth(0)
axes[1,0].xaxis.set_tick_params(width=2)
axes[1,0].yaxis.set_tick_params(width=2)


for axis in ['bottom','left']:
  axes[1,1].spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes[1,1].spines[axis].set_linewidth(0)
axes[1,1].xaxis.set_tick_params(width=2)
axes[1,1].yaxis.set_tick_params(width=2)


#axes[0].set_title("Forward Pass")
axes[1,0].set_yscale('linear')
axes[1,0].get_yaxis().get_major_formatter().set_useOffset(False)
#axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
#axes[0].ticklabel_format(axis='y', style='sci')
axes[1,0].plot(np.arange(2,13,1), ds1_forward*10**6,  linewidth=2.5, label="Func-RowCol", color=func_rowcol_color)
axes[1,0].plot(np.arange(2,13,1), ds2_forward*10**6,  linewidth=2.5, label="PB-BMP", color=pb_bmp_color)
axes[1,0].plot(np.arange(2,13,1), ds3_forward*10**6, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
#axes[0].plot(np.arange(2,13,1), ds4_forward,  linewidth=2.5, label="Func-BMP", color=func_bmp_color)
#axes[0].plot(np.arange(2,13,1), ds5_forward,  linewidth=2.5, label="Func-Pure", color=func_pure_color)

axes[1,0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
axes[1,0].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[1,0].set_ylabel("Energy (mJ)", fontweight='bold')


#axes[1].set_title("Backward Pass")
axes[1,1].set_yscale('linear')
#axes[1].get_yaxis().get_major_formatter().set_useOffset(False)
#axes[1].ticklabel_format(axis='y', style='sci')
axes[1,1].plot(np.arange(2,13,1), ds1_backward*10**6, linewidth=2.5, label="Func-RowCol", color=func_rowcol_color)
axes[1,1].plot(np.arange(2,13,1), ds2_backward*10**6, linewidth=2.5, label="PB-BMP", color=pb_bmp_color)
axes[1,1].plot(np.arange(2,13,1), ds3_backward*10**6, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
#axes[1].plot(np.arange(2,13,1), ds4_backward, linewidth=2.5, label="Func-BMP", color=func_bmp_color)
#axes[1].plot(np.arange(2,13,1), ds5_backward, linewidth=2.5, label="Func-Pure", color=func_pure_color)
#axes[1].legend(frameon=False)
axes[1,1].set_xlabel("Quantization Bits Weights", fontweight='bold') #, fontsize=14, fontweight='bold'
axes[1,1].set_ylabel("Energy (mJ)", fontweight='bold')


plt.tight_layout(pad=.1)
plt.savefig('/Users/clemens/Desktop/comb.png')
#plt.show()


# ####
# #### sparsity
# ####
# data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')

# # conv
# test = data[data['Layer'] == "FullyConnected"]
# test = test[test['Channels'] == 64]
# test = test[test['Filter Size'] == 3]
# test = test[test['Image Size'] == 28]
# test = test[test['Weight Bits'] == 8]
# test = test.drop_duplicates()
# test = test.sort_values(by=['Sparsity'])

# ds1 = test[test['Data Structure'] == 'Crossbar']
# ds1 = ds1.drop_duplicates()
# ds1_forward = ds1['Forward Energy']
# ds1_backward = ds1['Backward Energy'] + ds1['Update Energy']


# ds2 = test[test['Data Structure'] == 'PB-BMP']
# ds2 = ds2.drop_duplicates()
# ds2_forward = ds2['Forward Energy']
# ds2_backward = ds2['Backward Energy'] + ds2['Update Energy']


# ds3 = test[test['Data Structure'] == 'PB-CSR']
# ds3 = ds3.drop_duplicates()
# ds3_forward = ds3['Forward Energy']
# ds3_backward = ds3['Backward Energy'] + ds3['Update Energy']


# plt.clf()



# #plt.rc('font', family='serif')
# plt.rc('font', weight='bold')
# fig, axes = plt.subplots(nrows=2, ncols=1)

# #plt.rcParams['axes.labelsize'] = 16
# #plt.rcParams['axes.labelweight'] = 'bold'

# for axis in ['bottom','left']:
#   axes[0].spines[axis].set_linewidth(3)
# for axis in ['top','right']:
#   axes[0].spines[axis].set_linewidth(0)
# axes[0].xaxis.set_tick_params(width=2)
# axes[0].yaxis.set_tick_params(width=2)


# for axis in ['bottom','left']:
#   axes[1].spines[axis].set_linewidth(3)
# for axis in ['top','right']:
#   axes[1].spines[axis].set_linewidth(0)
# axes[1].xaxis.set_tick_params(width=2)
# axes[1].yaxis.set_tick_params(width=2)


# #axes[0].set_title("Forward Pass")
# axes[0].plot(np.arange(.05,1,.05), ds1_forward,  linewidth=2.5, label="Crossbar", color=crossbar_color)
# axes[0].plot(np.arange(.05,1,.05), ds2_forward,  linewidth=2.5, label="PB-BMP", color=pb_bmp_color)
# axes[0].plot(np.arange(.05,1,.05), ds3_forward, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
# axes[0].legend(frameon=False)
# axes[0].set_xlabel("Sparsity", fontweight='bold') #, fontsize=14, fontweight='bold'
# axes[0].set_ylabel("Energy (J)", fontweight='bold')


# #axes[1].set_title("Backward Pass")
# axes[1].plot(np.arange(.05,1,.05), ds1_backward, linewidth=2.5, label="Crossbar", color=crossbar_color)
# axes[1].plot(np.arange(.05,1,.05), ds2_backward, linewidth=2.5, label="PB-BMP", color=pb_bmp_color)
# axes[1].plot(np.arange(.05,1,.05),ds3_backward, linewidth=2.5, label="PB-CSR", color=pb_csr_color)
# axes[1].legend(frameon=False)
# axes[1].set_xlabel("Sparsity", fontweight='bold') #, fontsize=14, fontweight='bold'
# axes[1].set_ylabel("Energy (J)", fontweight='bold')


# plt.tight_layout()
# plt.savefig('/Users/clemens/Desktop/sparsity.png')



# IMSL2019

