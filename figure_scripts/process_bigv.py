import pickle
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join



onlyfiles = [f for f in listdir("results") if isfile(join("results", f))]

pickle_files = []
std_dev = set()
for i in onlyfiles:
	if ".pkl" in i:
		pickle_files.append(i)
	std_dev.add(i.split('_')[-2].split('2019-')[0])

std_dev = [float(i) for i in std_dev]
#std_dev = list(std_dev)



def get_ts(pickle_files, std_dev, ds_name, neuron_name):
	last_test = dict((el,[]) for el in list(std_dev))
	for i in pickle_files:
		if (ds_name in i) and (neuron_name in i):
			with open('results/'+i, 'rb') as f:
				results = pickle.load(f)
			last_test[float(i.split('_')[-2].split('2019-')[0])].append(results['test'][-1])

	x_LIF_DVS = np.arange(len(last_test))
	y_LIF_DVS = []
	y_median = []
	yerr_LIF_DVS = []
	y_LIF_DVS_joshi = []
	y_median_joshi = []
	yerr_LIF_DVS_joshi = []
	for key, values in sorted(last_test.items()):
		y_LIF_DVS.append(np.mean(values))
		y_median.append(np.median(values))
		yerr_LIF_DVS.append(np.std(values))


		# outlier cleaning
		values_reduce = values
		values_reduce.remove(max(values_reduce))
		values_reduce.remove(min(values_reduce))
		y_LIF_DVS_joshi.append(np.mean(values_reduce))
		y_median_joshi.append(np.median(values_reduce))
		yerr_LIF_DVS_joshi.append(np.std(values_reduce))


		# percentile
		#values_reduce = values
		#values_reduce.remove(max(values_reduce))
		#values_reduce.remove(min(values_reduce))
		#y_LIF_DVS_joshi.append(np.mean(values_reduce))
		#y_median_joshi.append(np.median(values_reduce))
		#yerr_LIF_DVS_joshi.append(np.std(values_reduce))

	return x_LIF_DVS, y_LIF_DVS, yerr_LIF_DVS, y_median, y_LIF_DVS_joshi, yerr_LIF_DVS_joshi, y_median_joshi


x_LIF_DVS, y_LIF_DVS , yerr_LIF_DVS, median_LIF_DVS, y_LIF_DVS_joshi, yerr_LIF_DVS_joshi, median_LIF_DVS_joshi = get_ts(pickle_files, std_dev, "_DVS_", "_LIF_")
x_LIF_FMNIST, y_LIF_FMNIST , yerr_LIF_FMNIST, median_LIF_FMNIST, y_LIF_FMNIST_joshi, yerr_LIF_FMNIST_joshi, median_LIF_FMNIST_joshi = get_ts(pickle_files, std_dev, "_FMNIST_", "_LIF_")
x_adexLIF_DVS, y_adexLIF_DVS , yerr_adexLIF_DVS, median_adexLIF_DVS, y_adexLIF_DVS_joshi , yerr_adexLIF_DVS_joshi, median_adexLIF_DVS_joshi = get_ts(pickle_files, std_dev, "_DVS_", "_adex_LIF_")
x_adexLIF_FMNIST, y_adexLIF_FMNIST , yerr_adexLIF_FMNIST, median_adexLIF_FMNIST,y_adexLIF_FMNIST_joshi , yerr_adexLIF_FMNIST_joshi, median_adexLIF_FMNIST_joshi = get_ts(pickle_files, std_dev, "_FMNIST_", "_adex_LIF_")





# mean all
plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Noise Level')

plt.errorbar(x_LIF_DVS, y_LIF_DVS , yerr=yerr_LIF_DVS, label='LIF DVS')
plt.errorbar(x_LIF_FMNIST, y_LIF_FMNIST , yerr=yerr_LIF_FMNIST, label='LIF FMNIST')
plt.errorbar(x_adexLIF_DVS, y_adexLIF_DVS , yerr=yerr_adexLIF_DVS, label='adexLIF DVS')
plt.errorbar(x_adexLIF_FMNIST, y_adexLIF_FMNIST , yerr=yerr_adexLIF_FMNIST, label='adexLIF FMNIST')

plt.legend(loc = 'best')
plt.title("Effect of Noise All")

plt.tight_layout()
plt.savefig("figures/effect_noise_presi.png")


# mean all joshi
plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Noise Level')

plt.errorbar(x_LIF_DVS, median_LIF_DVS_joshi , yerr=yerr_LIF_DVS_joshi, label='LIF DVS')
plt.errorbar(x_LIF_FMNIST, median_LIF_FMNIST_joshi , yerr=yerr_LIF_FMNIST_joshi, label='LIF FMNIST')
plt.errorbar(x_adexLIF_DVS, median_adexLIF_DVS_joshi , yerr=yerr_adexLIF_DVS_joshi, label='adexLIF DVS')
plt.errorbar(x_adexLIF_FMNIST, median_adexLIF_FMNIST_joshi , yerr=yerr_adexLIF_FMNIST_joshi, label='adexLIF FMNIST')

plt.legend(loc = 'best')
plt.title("Effect of Noise All Median (rem. min. max.)")

plt.tight_layout()
plt.savefig("figures/effect_noise_presi_median.png")


# mean all joshi
plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Noise Level')

plt.errorbar(x_LIF_DVS, y_LIF_DVS_joshi , yerr=yerr_LIF_DVS_joshi, label='LIF DVS')
plt.errorbar(x_LIF_FMNIST, y_LIF_FMNIST_joshi , yerr=yerr_LIF_FMNIST_joshi, label='LIF FMNIST')
plt.errorbar(x_adexLIF_DVS, y_adexLIF_DVS_joshi , yerr=yerr_adexLIF_DVS_joshi, label='adexLIF DVS')
plt.errorbar(x_adexLIF_FMNIST, y_adexLIF_FMNIST_joshi , yerr=yerr_adexLIF_FMNIST_joshi, label='adexLIF FMNIST')

plt.legend(loc = 'best')
plt.title("Effect of Noise All Mean (rem. min. max.)")

plt.tight_layout()
plt.savefig("figures/effect_noise_presi_mean.png")


# median LIF
plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Noise Level')

plt.errorbar(x_LIF_DVS, y_LIF_DVS , yerr=yerr_LIF_DVS, label='LIF DVS mean')
plt.errorbar(x_LIF_FMNIST, y_LIF_FMNIST , yerr=yerr_LIF_FMNIST, label='LIF FMNIST mean')
plt.plot(x_LIF_DVS, median_LIF_DVS , label='LIF DVS median')
plt.plot(x_adexLIF_FMNIST, median_LIF_FMNIST ,  label='LIF FMNIST median')

plt.legend(loc = 'best')
plt.title("Effect of Noise All")

plt.tight_layout()
plt.savefig("figures/effect_noise_all_median.png")

plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Noise Level')

plt.errorbar(x_LIF_DVS, y_LIF_DVS , yerr=yerr_LIF_DVS, label='LIF DVS')
#plt.errorbar(x_LIF_FMNIST, y_LIF_FMNIST , yerr=yerr_LIF_FMNIST, label='LIF FMNIST')
#plt.errorbar(x_adexLIF_DVS, y_adexLIF_DVS , yerr=yerr_adexLIF_DVS, label='adexLIF DVS')
#plt.errorbar(x_adexLIF_FMNIST, y_adexLIF_FMNIST , yerr=yerr_adexLIF_FMNIST, label='adexLIF FMNIST')

plt.legend(loc = 'best')
plt.title("Effect of Noise LIF DVS")

plt.tight_layout()
plt.savefig("figures/effect_noiseLD.png")



plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Noise Level')

#plt.errorbar(x_LIF_DVS, y_LIF_DVS , yerr=yerr_LIF_DVS, label='LIF DVS')
plt.errorbar(x_LIF_FMNIST, y_LIF_FMNIST , yerr=yerr_LIF_FMNIST, label='LIF FMNIST')
#plt.errorbar(x_adexLIF_DVS, y_adexLIF_DVS , yerr=yerr_adexLIF_DVS, label='adexLIF DVS')
#plt.errorbar(x_adexLIF_FMNIST, y_adexLIF_FMNIST , yerr=yerr_adexLIF_FMNIST, label='adexLIF FMNIST')

plt.legend(loc = 'best')
plt.title("Effect of Noise LIF FMNIST")

plt.tight_layout()
plt.savefig("figures/effect_noiseLF.png")


plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Noise Level')

#plt.errorbar(x_LIF_DVS, y_LIF_DVS , yerr=yerr_LIF_DVS, label='LIF DVS')
#plt.errorbar(x_LIF_FMNIST, y_LIF_FMNIST , yerr=yerr_LIF_FMNIST, label='LIF FMNIST')
plt.errorbar(x_adexLIF_DVS, y_adexLIF_DVS , yerr=yerr_adexLIF_DVS, label='adexLIF DVS')
#plt.errorbar(x_adexLIF_FMNIST, y_adexLIF_FMNIST , yerr=yerr_adexLIF_FMNIST, label='adexLIF FMNIST')

plt.legend(loc = 'best')
plt.title("Effect of Noise adexLIF DVS")

plt.tight_layout()
plt.savefig("figures/effect_noiseAD.png")



plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('Noise Level')

#plt.errorbar(x_LIF_DVS, y_LIF_DVS , yerr=yerr_LIF_DVS, label='LIF DVS')
#plt.errorbar(x_LIF_FMNIST, y_LIF_FMNIST , yerr=yerr_LIF_FMNIST, label='LIF FMNIST')
#plt.errorbar(x_adexLIF_DVS, y_adexLIF_DVS , yerr=yerr_adexLIF_DVS, label='adexLIF DVS')
plt.errorbar(x_adexLIF_FMNIST, y_adexLIF_FMNIST , yerr=yerr_adexLIF_FMNIST, label='adexLIF FMNIST')

plt.legend(loc = 'best')
plt.title("Effect of Noise adexLIF FMNIST")

plt.tight_layout()
plt.savefig("figures/effect_noiseAF.png")