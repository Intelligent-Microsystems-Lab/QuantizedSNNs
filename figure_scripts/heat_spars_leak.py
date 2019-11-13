import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import matplotlib as mpl

mpl.rc('image', cmap='rainbow')

crossbar_color = "r"
pb_bmp_color = "k"
pb_csr_color = "b"
func_rowcol_color = "tab:orange"
func_bmp_color = "m"
func_pure_color = "g"

data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')
test = data[data['Layer'] == "FullyConnected"]
test = test[test['Channels'] == 64]
test = test[test['Filter Size'] == 3]
test = test[test['Image Size'] == 28]
test = test[test['Weight Bits'] == 8]
test = test.drop_duplicates()
sparse_levels = np.unique(test['Sparsity']) 


def comput_sparse_leak_mat(ds_struct):

	

	test = data[data['Layer'] == "FullyConnected"]
	test = test[test['Channels'] == 64]
	test = test[test['Filter Size'] == 3]
	test = test[test['Image Size'] == 28]
	test = test[test['Weight Bits'] == 8]
	test = test.drop_duplicates()

	#ds1 = test[test['Data Structure'] == 'Crossbar']
	#ds1 = ds1.drop_duplicates()
	

	ener_mat = np.zeros([len(sparse_levels), len(np.arange(.0,1,.05))])
	for s_idx, sparse_level in enumerate(sparse_levels):
		test_t = test[test['Sparsity'] == sparse_level]
		ds1 = test_t[test_t['Data Structure'] == ds_struct]
		ds1 = ds1.drop_duplicates()
		for i_idx, i in enumerate(np.arange(.0,1,.05)):
			fixed = float( ds1['Forward Energy'] + ds1['Backward Energy'] )
			variable_fac = float(ds1['Memory Power'])
			mul_fact = (fixed*i)/(variable_fac-variable_fac*i)
			ener_mat[s_idx, i_idx] = fixed + variable_fac * mul_fact 
	return ener_mat


ener_mat1 = comput_sparse_leak_mat("Crossbar")
ener_mat2 = comput_sparse_leak_mat("PB-BMP")
ener_mat3 = comput_sparse_leak_mat("PB-CSR")


ds1_x = []
ds1_y = []
ds1_c = []

ds2_x = []
ds2_y = []
ds2_c = []

ds3_x = []
ds3_y = []
ds3_c = []

for s_idx, sparse_level in enumerate(sparse_levels):
	for i_idx, i in enumerate(np.arange(.0,1,.05)):
		if (ener_mat1[s_idx, i_idx] < ener_mat2[s_idx, i_idx]) and (ener_mat1[s_idx, i_idx] < ener_mat3[s_idx, i_idx]):
			ds1_x.append(sparse_level)
			ds1_y.append(i)
			ds1_c.append(ener_mat1[s_idx, i_idx])
		if (ener_mat2[s_idx, i_idx] < ener_mat1[s_idx, i_idx]) and (ener_mat2[s_idx, i_idx] < ener_mat3[s_idx, i_idx]):
			ds2_x.append(sparse_level)
			ds2_y.append(i)
			ds2_c.append(ener_mat2[s_idx, i_idx])
		if (ener_mat3[s_idx, i_idx] < ener_mat1[s_idx, i_idx]) and (ener_mat3[s_idx, i_idx] < ener_mat2[s_idx, i_idx]):
			ds3_x.append(sparse_level)
			ds3_y.append(i)
			ds3_c.append(ener_mat3[s_idx, i_idx])




plt.clf()

#plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=1, ncols=1)

#plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)


axes.set_yscale('linear')
axes.set_xscale('linear')

ds_x_cum = ds1_x + ds2_x + ds3_x
ds_y_cum = ds1_y + ds2_y + ds3_y
ds_c_cum = ds1_c + ds2_c + ds3_c

#sc1_cum = axes.scatter(ds_x_cum, ds_y_cum, c = np.log(ds_c_cum), marker = ",")
sc1 = axes.scatter(ds1_x, ds1_y, c = np.log(ds1_c), marker = "x", label="Crossbar")
sc2 = axes.scatter(ds2_x, ds2_y, c = np.log(ds2_c), marker = "s",label="PB-BMP")
sc3 = axes.scatter(ds3_x, ds3_y, c = np.log(ds3_c), marker = "o",label="PB-CSR")
leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False)
axes.set_xlabel("Density", fontweight='bold') #, fontsize=14, fontweight='bold'
axes.set_ylabel("Leakage Contribution", fontweight='bold')
#plt.cm.seismic

leg.legendHandles[0].set_color(crossbar_color)
leg.legendHandles[1].set_color(pb_bmp_color)
leg.legendHandles[2].set_color(pb_csr_color)


cbar = plt.colorbar(sc1)
cbar.ax.set_xlabel('Orders of\nMagnitude\nJoules')
#cbar.ax.set_ticks_position('top')
plt.tight_layout()
plt.savefig('/Users/clemens/Desktop/heat.png')



# ####
# #### alpha sweep plot
# ####
# data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')

# # conv
# test = data[data['Layer'] == "FullyConnected"]
# test = test[test['Channels'] == 64]
# test = test[test['Filter Size'] == 3]
# test = test[test['Image Size'] == 28]
# test = test[test['Weight Bits'] == 2]
# test = test[test['Sparsity'] == .25]
# test = test.drop_duplicates()

# ds1 = test[test['Data Structure'] == 'Crossbar']
# ds1 = ds1.drop_duplicates()
# ds1_forward = []
# for i in np.arange(.0,1,.01):
# 	fixed = float( ds1['Forward Energy'] + ds1['Backward Energy'] + ds1['Update Energy'])
# 	variable_fac = float(ds1['Memory Power'])
# 	mul_fact = (fixed*i)/(variable_fac-variable_fac*i)
# 	ds1_forward.append(fixed + variable_fac * mul_fact )


# ds2 = test[test['Data Structure'] == 'PB-BMP']
# ds2 = ds2.drop_duplicates()
# ds2_forward = []
# for i in np.arange(.0,1,.01):
# 	fixed = float( ds2['Forward Energy'] + ds2['Backward Energy'] + ds2['Update Energy'])
# 	variable_fac = float(ds2['Memory Power'])
# 	mul_fact = (fixed*i)/(variable_fac-variable_fac*i)
# 	ds2_forward.append(fixed + variable_fac * mul_fact )


# ds3 = test[test['Data Structure'] == 'PB-CSR']
# ds3 = ds3.drop_duplicates()
# ds3_forward = []
# for i in np.arange(.0,1,.01):
# 	fixed = float( ds3['Forward Energy'] + ds3['Backward Energy'] + ds3['Update Energy'])
# 	variable_fac = float(ds3['Memory Power'])
# 	mul_fact = (fixed*i)/(variable_fac-variable_fac*i)
# 	ds3_forward.append(fixed + variable_fac * mul_fact )



# data = pd.read_csv('/Users/clemens/Desktop/patrick_results.csv')

# test = data[data['Layer'] == "FullyConnected"]
# test = test[test['Channels'] == 64]
# test = test[test['Filter Size'] == 3]
# test = test[test['Image Size'] == 28]
# test = test[test['Weight Bits'] == 8]
# test = test[test['Sparsity'] == .25]
# test = test.drop_duplicates()

# ds1 = test[test['Data Structure'] == 'Crossbar']
# ds1 = ds1.drop_duplicates()
# ds18_forward = []
# for i in np.arange(.0,1,.01):
# 	fixed = float( ds1['Forward Energy'] + ds1['Backward Energy'] + ds1['Update Energy'])
# 	variable_fac = float(ds1['Memory Power'])
# 	mul_fact = (fixed*i)/(variable_fac-variable_fac*i)
# 	ds18_forward.append(fixed + variable_fac * mul_fact )


# ds2 = test[test['Data Structure'] == 'PB-BMP']
# ds2 = ds2.drop_duplicates()
# ds28_forward = []
# for i in np.arange(.0,1,.01):
# 	fixed = float( ds2['Forward Energy'] + ds2['Backward Energy'] + ds2['Update Energy'])
# 	variable_fac = float(ds2['Memory Power'])
# 	mul_fact = (fixed*i)/(variable_fac-variable_fac*i)
# 	ds28_forward.append(fixed + variable_fac * mul_fact )


# ds3 = test[test['Data Structure'] == 'PB-CSR']
# ds3 = ds3.drop_duplicates()
# ds38_forward = []
# for i in np.arange(.0,1,.01):
# 	fixed = float( ds3['Forward Energy'] + ds3['Backward Energy'] + ds3['Update Energy'])
# 	variable_fac = float(ds3['Memory Power'])
# 	mul_fact = (fixed*i)/(variable_fac-variable_fac*i)
# 	ds38_forward.append(fixed + variable_fac * mul_fact )


# plt.clf()

# #plt.rc('font', family='serif')
# plt.rc('font', weight='bold')
# fig, axes = plt.subplots(nrows=1, ncols=1)

# #plt.rcParams['axes.labelsize'] = 16
# #plt.rcParams['axes.labelweight'] = 'bold'

# for axis in ['bottom','left']:
#   axes.spines[axis].set_linewidth(3)
# for axis in ['top','right']:
#   axes.spines[axis].set_linewidth(0)
# axes.xaxis.set_tick_params(width=2)
# axes.yaxis.set_tick_params(width=2)


# axes.set_yscale('log')
# axes.set_xscale('linear')
# #axes[0].set_title("Forward Pass")
# axes.plot(np.arange(.0,1,.01), ds1_forward,  linewidth=2.5, label="Crossbar 2bit", color=crossbar_color)
# axes.plot(np.arange(.0,1,.01), ds2_forward,  linewidth=2.5, label="PB-BMP 2bit", color=pb_bmp_color)
# axes.plot(np.arange(.0,1,.01), ds3_forward, linewidth=2.5, label="PB-CSR 2bit", color=pb_csr_color)

# axes.plot(np.arange(.0,1,.01), ds18_forward,  linewidth=2.5, linestyle='dashed', label="Crossbar 8bit", color=crossbar_color)
# axes.plot(np.arange(.0,1,.01), ds28_forward,  linewidth=2.5, linestyle='dashed', label="PB-BMP 8bit", color=pb_bmp_color)
# axes.plot(np.arange(.0,1,.01), ds38_forward, linewidth=2.5, linestyle='dashed', label="PB-CSR 8bit", color=pb_csr_color)
# axes.legend(frameon=False)
# axes.set_xlabel("Leakage Contribution", fontweight='bold') #, fontsize=14, fontweight='bold'
# axes.set_ylabel("Energy (J)", fontweight='bold')


# plt.tight_layout()
# plt.savefig('/Users/clemens/Desktop/fc_sweep.png')

