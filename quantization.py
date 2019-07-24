import numpy as np
import torch



def normalize_distribution(mu, var):
	new_mu = mu/mu.max()
	new_v = new_mu/(mu/var)
	return new_mu, new_v


def quantize(weights, mu, var, device):
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










# test routine
#weights = [torch.Tensor([-.8, .7, 5, -.002]), torch.Tensor([-.8, .7, 5, -.002])]

#mu = torch.Tensor([-1.1, -.61, -.29, .002, .40, .74, 1.2]) #3 bit precision
#var = torch.Tensor([.01, .009, .008, .01, .009, .008, .01])

#n_mu, n_var = normalize_distribution(mu, var)
#w = quantize(weights, n_mu, n_var)
