import numpy as np
import torch



def normalize_distribution(mu, var):
	new_mu = 2*(mu-mu.min())/(mu.max()-mu.min())-1
	new_v = abs(new_mu)/(abs(mu)/var)
	return new_mu, new_v



def quantize(weights, mu, var):
	m = torch.distributions.normal.Normal(mu, var)

	for i,layer_w in enumerate(weights):
		dim = layer_w.shape
		layer_w = layer_w.flatten()
		m_temp = m.sample([layer_w.shape[0]])
		temp_diff = torch.abs(m_temp - layer_w[:,None])
		_, ind_m = temp_diff.min(dim=1)
		weights[i] = torch.gather(input = m_temp, dim = 1, index = ind_m.view(-1,1)).reshape(dim)
		weights[i].requires_grad = True

	return weights


#        (b-a)(x - min)
# f(x) = --------------  + a
#           max - min

# mu1 = torch.Tensor([.1022, .368, .822, 1.36, 1.95, 2.55, 2.92, 3.36])*10e-6

# diff1 = mu1[1:] - mu1[:-1]
# diff2 = mu2[1:] - mu2[:-1]

# mu = torch.Tensor([.254, .589, .997, 1.3, 1.72, 2.24, 2.8, 3.36]).to(device)*10e-6
# var = torch.Tensor([5.8, 4.92, 5.91, 5.91, 7.57, 10.9, 12.1, 12.5]).to(device)*10e-8

# new_mu, new_v = normalize_distribution(mu, var)

# test routine
#weights = [torch.Tensor([-.8, .7, 5, -.002]), torch.Tensor([-.8, .7, 5, -.002])]

#mu = torch.Tensor([-1.1, -.61, -.29, .002, .40, .74, 1.2]) #3 bit precision
#var = torch.Tensor([.01, .009, .008, .01, .009, .008, .01])

#n_mu, n_var = normalize_distribution(mu, var)
#w = quantize(weights, n_mu, n_var)
