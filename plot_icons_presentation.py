import matplotlib.pyplot as plt
import numpy as np




plt.clf()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')
plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=1, ncols=1) #
for axis in ['bottom','left']:
    axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
    axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)
 
year = [2016, 2015, 2017, 2015, 2016, 2020]
acc = [75.42, 77.43, 90.85, 82.95, 87.5, 91.24]
author = ['Panda 2016', 'Cao 2015', 'Rueckauer 2017', 'Hunsberger 2015', 'Esser 2016', 'Wu 2020']

ann_x = [2014, 2015, 2016, 2017, 2018, 2019]
ann_y = [91.20, 96.5, 96.54, 97.8, 99.0, 99.0]


axes.scatter(year, acc, label = 'SNNs', marker = 'x')

axes.plot(ann_x, ann_y, '--', label = 'ANNs')


for i, txt in enumerate(author):
    axes.annotate(txt, (year[i]+.1, acc[i]+.05), fontsize = 11)


# Add xticks on the middle of the group bars
axes.set_xlabel('Year', fontweight='bold')
axes.set_ylabel('Accuracy', fontweight='bold')
#axes.legend(loc='upper center', bbox_to_anchor=(0.5, -.15), ncol=4, frameon=False)
#axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
plt.title("State-of-the-Art CIFAR-10")

axes.set_ylim([55, 100])

plt.legend(frameon = False, loc = 'lower right')
plt.tight_layout()
plt.savefig("Scatter.pdf")




import matplotlib.pyplot as plt
import numpy as np


s_in = np.zeros(1000) 
s_in[1::100] = 1

s_in[420] = 1
s_in[430] = 1
q_v = np.zeros(1000) 
p_v = np.zeros(1000) 
u_v = np.zeros(1000) 
r_v = np.zeros(1000) 
s_out = np.zeros(1000)

for i in range(len(s_in)-1):
	q_v[i+1] = 0.85 * q_v[i] + s_in[i+1]
	p_v[i+1] = 0.75 * p_v[i] + q_v[i+1]
	s_out[i+1] = float(p_v[i+1] - r_v[i] > 2.5)
	r_v[i+1] = 0.75 * r_v[i] + s_out[i+1]


plt.clf()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='18')
plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='16')

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4, 4.8))
for i in range(4): #
	for axis in ['bottom','left']:
	    axes[i].spines[axis].set_linewidth(2)
	for axis in ['top','right']:
	    axes[i].spines[axis].set_linewidth(0)
	axes[i].xaxis.set_tick_params(width=2)
	axes[i].yaxis.set_tick_params(width=2)

axes[0].xaxis.set_ticklabels([])
axes[1].xaxis.set_ticklabels([])
axes[2].xaxis.set_ticklabels([])


axes[0].plot(s_in[250:650], color = 'k')
axes[1].plot(q_v[250:650], color = 'k')
axes[2].plot(p_v[250:650], color = 'k')
axes[2].plot(np.ones(1000)[250:650]*2.5, '--', color = 'k')
axes[3].plot(r_v[250:650], color = 'k')

axes[0].set_ylabel("S[n]")
axes[1].set_ylabel("Q[n]")
axes[2].set_ylabel("P[n]")
axes[3].set_ylabel("R[n]")

#axes[0].set_title("Neural Dynamics")

plt.tight_layout()
plt.savefig("UPQR.pdf")




import matplotlib.pyplot as plt
import numpy as np

deriv = np.zeros(1000)
deriv[500] = 1
x = np.arange(-10,10,.1)        
sur = (1/(1 + np.exp(-x))) * (1 - 1/(1 + np.exp(-x))) 

plt.clf()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='18')
plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='16')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.4, 3.2))
for i in range(2): #
	for axis in ['bottom','left']:
	    axes[i].spines[axis].set_linewidth(2)
	for axis in ['top','right']:
	    axes[i].spines[axis].set_linewidth(0)
	axes[i].xaxis.set_tick_params(width=2)
	axes[i].yaxis.set_tick_params(width=2)

axes[0].xaxis.set_ticklabels([])
axes[1].xaxis.set_ticklabels([])
axes[0].yaxis.set_ticklabels([])
axes[1].yaxis.set_ticklabels([])

axes[0].plot(deriv, color = 'k')
axes[1].plot(sur, color = 'k')

axes[0].set_ylabel(r"$\frac{\partial}{\partial U}S$")
axes[1].set_ylabel(r"$\sigma'$")
axes[0].set_xlabel("U[n]")
axes[1].set_xlabel("U[n]")


#axes[0].set_title("Neural Dynamics")

plt.tight_layout()
plt.savefig("sur.pdf")
