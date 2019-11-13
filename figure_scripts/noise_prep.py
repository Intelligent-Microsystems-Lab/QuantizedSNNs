import torch
import matplotlib.pyplot as plt

nb = 700
time_steps = 250

m = torch.distributions.poisson.Poisson(75)
test = np.cumsum(m.sample(sample_shape=torch.Size([100])))
valid_ind = test[test < 250]

cur_rate = 1

mat_rec = []
for idx, i in enumerate(range(nb)):
	if idx%5 == 0:
		cur_rate += .25
	m = torch.distributions.poisson.Poisson(cur_rate)
	test = np.cumsum(m.sample(sample_shape=torch.Size([10000])))
	valid_ind = test[test < 250]
	temp = torch.zeros([time_steps])
	for i in  valid_ind:
		temp[int(i)] = 1

	mat_rec.append(temp)

final = torch.stack(mat_rec,dim=0)

plt.imshow(final)
plt.show()


with open('/Users/clemens/Desktop/input_700_250_25.pkl', 'wb') as f:
    pickle.dump(final, f)

