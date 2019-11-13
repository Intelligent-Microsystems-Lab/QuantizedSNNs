import imageio
import numpy as np
import matplotlib.pyplot as plt
import pickle

#with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_nd_precise.pkl", 'rb') as f:
#    output = pickle.load(f)

#plt.imshow(output['output'].detach().numpy())
#plt.show()


wb_sweep = [2,3,4,5,6]
ab_sweep = [6,7,8,10,12]
gb_sweep = [8,9,10,11,12]


trial_list = ['','_t0','_t1','_t2','_t3',]

lr_cur = 1.

print("lr 1")

# print("ab")
# prob_list = []
# #ab sweep
# for wb_cur in wb_sweep:
# 	print(str(wb_cur)+"", end="")
# 	avg_reach = []
# 	for ab_cur in ab_sweep:
# 		avg_list = []
# 		reach_list = []
# 		for trial in trial_list:
# 			bit_string = str(wb_cur) + str(ab_cur) + str(8) + str(ab_cur) + "_" + str(lr_cur)+trial
# 			try:
# 				with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string+".pkl", 'rb') as f:
# 					results = pickle.load(f)
# 				avg_list.append(np.min(results['loss_hist']))
# 				reach_list.append(np.argmin(results['loss_hist']))
# 			except:
# 				prob_list.append(bit_string)
# 		print(" & " +str(np.round(np.average(avg_list),2))+" \\textit{$\\pm"+str(np.round(np.std(avg_list), 2))+"$}", end="")
# 		avg_reach.append(str(np.round(np.average(reach_list),2)))
# 	print("\\\\")
# 	for ep_cur in avg_reach:
# 		print(" & ("+ep_cur +") ", end="")
# 	print("\\\\")





print("gb")
prob_list = []
#ab sweep
for wb_cur in wb_sweep:
	print(str(wb_cur)+"", end="")
	avg_reach = []
	for gb_cur in gb_sweep:
		avg_list = []
		reach_list = []
		for trial in trial_list:
			bit_string = str(wb_cur) + str(12) + str(gb_cur) + str(12) + "_" + str(lr_cur)+trial
			try:
				with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string+".pkl", 'rb') as f:
					results = pickle.load(f)
				avg_list.append(np.min(results['loss_hist']))
				reach_list.append(np.argmin(results['loss_hist']))
			except:
				prob_list.append(bit_string)
		print(" & " +str(int(np.round(np.average(avg_list),0)))+" \\textit{$\\pm"+str(int(np.round(np.std(avg_list), 0)))+"$}", end="")
		avg_reach.append(str(int(np.round(np.average(reach_list),0))))
	print("\\\\")
	for ep_cur in avg_reach:
		print(" & ("+ep_cur +") ", end="")
	print("\\\\")

print(prob_list)
##############
##############
##############

# best epoch + max value which epoch

# def xover(ts, cut):
#     x = ts < cut
#     return x.argmax() if x.any() else -1

# lr_cur = .1

# print("lr .1")

# print("ab")
# #ab sweep
# for wb_cur in wb_sweep:
# 	print(str(wb_cur)+"", end="")
# 	for ab_cur in ab_sweep:
# 		bit_string = str(wb_cur) + str(ab_cur) + str(8) + str(ab_cur) + "_" + str(lr_cur)
# 		try:
# 			with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string+".pkl", 'rb') as f:
# 				results = pickle.load(f)
					
# 			print(" & " +str(np.argmin(results['loss_hist']))+ " "+str(xover(np.array(results['loss_hist']), 6676.38)), end="")
# 		except:
# 			print(" & ????.?? ", end="")
# 	print("\\\\")


# print("gb")
# #ab sweep
# for wb_cur in wb_sweep:
# 	print(str(wb_cur)+"", end="")
# 	for ab_cur in gb_sweep:
# 		bit_string = str(wb_cur) + str(12) + str(ab_cur) + str(12) + "_" + str(lr_cur)
# 		try:
# 			with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string+".pkl", 'rb') as f:
# 				results = pickle.load(f)
# 			print(" & " +str(np.argmin(results['loss_hist']))+ " "+str(xover(np.array(results['loss_hist']), 6676.38)), end="")
# 		except:
# 			print(" & ????.?? ", end="")
# 	print("\\\\")


# lr_cur = 1.

# print("lr 1")

# print("ab")
# #ab sweep
# for wb_cur in wb_sweep:
# 	print(str(wb_cur)+"", end="")
# 	for ab_cur in ab_sweep:
# 		bit_string = str(wb_cur) + str(ab_cur) + str(8) + str(ab_cur) + "_" + str(lr_cur)
# 		try:
# 			with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string+".pkl", 'rb') as f:
# 				results = pickle.load(f)
# 			print(" & " +str(np.argmin(results['loss_hist']))+ " "+str(xover(np.array(results['loss_hist']), 6676.38)), end="")
# 		except:
# 			print(" & ????.?? ", end="")
# 	print("\\\\")


# print("gb")
# #ab sweep
# for wb_cur in wb_sweep:
# 	print(str(wb_cur)+"", end="")
# 	for ab_cur in gb_sweep:
# 		bit_string = str(wb_cur) + str(12) + str(ab_cur) + str(12) + "_" + str(lr_cur)
# 		try:
# 			with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string+".pkl", 'rb') as f:
# 				results = pickle.load(f)
# 			print(" & " +str(np.argmin(results['loss_hist']))+ " "+str(xover(np.array(results['loss_hist']), 6676.38)), end="")
# 		except:
# 			print(" & ????.?? ", end="")
# 	print("\\\\")


#####
#####
#####

# learning curve difference 2888

# bit_string1 = "4888_0.1"
# with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string1+".pkl", 'rb') as f:
# 	results1 = pickle.load(f)


# bit_string2 = "4888_1.0"
# with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string2+".pkl", 'rb') as f:
# 	results2 = pickle.load(f)




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



# axes.plot(results1['loss_hist'],  linewidth=2.5,  label="learning rate = 0.1")
# axes.plot(results2['loss_hist'],  linewidth=2.5, label="learning rate = 1.0")
# axes.legend(frameon=False)
# axes.set_xlabel("Epochs", fontweight='bold') #, fontsize=14, fontweight='bold'
# axes.set_ylabel("Van Rossum Distance", fontweight='bold')


# plt.tight_layout()
# plt.savefig('/Users/clemens/Desktop/vr_lr.png')


# lr_cur = .1

# print("lr .1")

# print("ab")
# #ab sweep
# for wb_cur in wb_sweep:
# 	print(str(wb_cur)+"", end="")
# 	for ab_cur in ab_sweep:
# 		bit_string = str(wb_cur) + str(ab_cur) + str(8) + str(ab_cur) + "_" + str(lr_cur)
# 		try:
# 			with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string+".pkl", 'rb') as f:
# 				results = pickle.load(f)
# 			print(" & " +str(np.round(np.min(results['loss_hist']),2)), end="")
# 		except:
# 			print(" & ????.?? ", end="")
# 	print("\\\\")


# print("gb")
# #ab sweep
# for wb_cur in wb_sweep:
# 	print(str(wb_cur)+"", end="")
# 	for ab_cur in gb_sweep:
# 		bit_string = str(wb_cur) + str(12) + str(ab_cur) + str(12) + "_" + str(lr_cur)
# 		try:
# 			with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string+".pkl", 'rb') as f:
# 				results = pickle.load(f)
# 			print(" & " +str(np.round(np.min(results['loss_hist']),2)), end="")
# 		except:
# 			print(" & ????.?? ", end="")
# 	print("\\\\")



# print("gb")
# #ab sweep
# for wb_cur in wb_sweep:
# 	print(str(wb_cur)+"", end="")
# 	for ab_cur in gb_sweep:
# 		bit_string = str(wb_cur) + str(12) + str(ab_cur) + str(12) + "_" + str(lr_cur)
# 		try:
# 			with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_smile_precise_"+bit_string+".pkl", 'rb') as f:
# 				results = pickle.load(f)
# 			print(" & " +str(np.round(np.min(results['loss_hist']),2)), end="")
# 		except:
# 			print(" & ????.?? ", end="")
# 	print("\\\\")

