import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd


with open("/Users/clemens/mnt/crc/QuantizedSNN/results/snn_illustrate_34343434_0.1.pkl", 'rb') as f:
	results = pickle.load(f)




plt.clf()
plt.rc('font', weight='bold')
fig, axes = plt.subplots(nrows=1, ncols=1)

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(3)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)



#axes[0].set_title("Forward Pass")
axes.plot(np.arange(6,6401,1), results['loss_hist'][5:6400],  linewidth=2.5, color="black")
axes.set_xlabel("Epochs", fontweight='bold') #, fontsize=14, fontweight='bold'
axes.set_ylabel("Van Rossum Distance", fontweight='bold')


plt.tight_layout()
#plt.savefig('/Users/clemens/Desktop/vr.png')
plt.show()


plt.clf()
plt.imshow(results['output_hist'][60].detach().numpy(),cmap='Greys')
#plt.axis('off')
#plt.savefig('/Users/clemens/Desktop/vr1.png')
plt.show()


plt.clf()
plt.imshow(results['output_hist'][600].detach().numpy(),cmap='Greys')
#plt.axis('off')
plt.savefig('/Users/clemens/Desktop/vr2.png')
plt.show()


plt.clf()
plt.imshow(results['output_hist'][2000].detach().numpy(),cmap='Greys')
#plt.axis('off')
#plt.savefig('/Users/clemens/Desktop/vr3.png')
plt.show()


plt.clf()
plt.imshow(results['output_hist'][6400].detach().numpy(),cmap='Greys')
#plt.axis('off')
#plt.savefig('/Users/clemens/Desktop/vr4.png')
plt.show()

#with open("/Users/clemens/mnt/crc/QuantizedSNN/data/nand70.pkl", 'rb') as f:
#    y_train = pickle.load(f)

#plt.imshow(y_train)
#plt.show()