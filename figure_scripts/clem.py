from energy_model import *
from itertools import product
import sys

#input = sys.argv
weight = 2 #int(input[2])
channels = 32 #int(input[4])
filter = 3 #int(input[6])
image = 28#int(input[8])
network = 'Convolution'# input[10]
data_structure = 'PB-CSR'#input[12]
sparsity = .1#float(input[14])
filename = f'OutLogs/{weight}_{channels}_{filter}_{image}_{network}_{data_structure}_{sparsity}.txt'

print(channels,filter,image,network,data_structure,sparsity)

quanty = Quantized_Network(weight,8,8,8,True)
if network == 'Convolution':
    quanty.network = [Convolution(channels,1,image,filter,1,sparsity,quanty.quantization,data_structure)]
else:
    quanty.network = [FullyConnected(image**2,512,sparsity,quanty.quantization,data_structure)]
network,ops,bits,energy,time,mem_power = quanty.ds_isolation()



output = [str(input[1:])+'\n',
          'Energy: '+str(energy)+'\n',
          'Time: '+str(time)+'\n',
          'Memory Power: '+str(mem_power)+'\n']

with open(filename,'w') as f:
    f.writelines(output)
