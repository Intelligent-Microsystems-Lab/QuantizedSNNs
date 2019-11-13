import numpy as np
import os

main_string1 = """#!/bin/csh

#$ -M cschaef6@nd.edu	 # Email address for job notification
#$ -m abe
#$ -q """

main_string2 = """# Specify queue (use ‘debug’ for development) // gpu@qa-2080ti-001.crc.nd.edu , qa-2080ti-002.crc.nd.edu , ta-titanv-001.crc.nd.edu, qa-titanx-001.crc.nd.edu, qa-1080ti-001.crc.nd.edu, da-1080-001.crc.nd.edu
#$ -l gpu_card=1 
#$ -N snn_smile_wage_"""
main_string3 = """     # Specify job name
#$ -o ./logs/output_snn_smile_"""
main_string4 = """_wage.txt
#$ -e ./logs/error_snn_smile_"""
main_string5 = """_wage.txt

module load python       # Required modules
setenv OMP_NUM_THREADS $NSLOTS

python spytorch_precise_01.py -wb """
main_string6 = """ -wb 2 -ab 6 -eb 6 -gb 8"""



wb_sweep = [2,3,4,5,6]
ab_sweep = [6,7,8,10,12]
gb_sweep = [8,9,10,11,12]


trials = 4
#bit_sweep = [ 4, 8, 16, 32, 2, 64]
#lr_sweep = [2,1,0.75,.5,.1,.01,.001,.0001,.00001]


# #ab sweep
# lr_cur = .1
# for w_cur in wb_sweep:
# 	for a_cur in ab_sweep:
# 		bit_string = str(w_cur) + str(a_cur) + "8" + str(a_cur) + "_" +str(lr_cur)
# 		file_string = main_string1 + "gpu" + main_string2 + bit_string + main_string3 + bit_string + main_string4 + bit_string + main_string5 + str(w_cur) + " -ab " + str(a_cur) + " -eb " + str(a_cur) + " -gb " + str(8) + " -lr " + str(lr_cur)
# 		with open('jobscripts/snn_01_'+bit_string+'.script', 'w') as f:
# 			f.write(file_string)
# 		os.system("qsub "+'jobscripts/snn_01_'+bit_string+'.script')


# #gb sweep
# for w_cur in wb_sweep:
# 	for a_cur in gb_sweep:
# 		bit_string = str(w_cur) + "12" + str(a_cur) + "12" + "_" +str(lr_cur)
# 		file_string = main_string1 + "gpu" + main_string2 + bit_string + main_string3 + bit_string + main_string4 + bit_string + main_string5 + str(w_cur) + " -ab " + str(12) + " -eb " + str(12) + " -gb " + str(a_cur) + " -lr " + str(lr_cur)
# 		with open('jobscripts/snn_'+bit_string+'.script', 'w') as f:
# 			f.write(file_string)
# 		os.system("qsub "+'jobscripts/snn_'+bit_string+'.script')


for trial in range(trials):

	#ab sweep
	lr_cur = 1
	for w_cur in wb_sweep:
		for a_cur in ab_sweep:
			bit_string = str(w_cur) + str(a_cur) + "8" + str(a_cur) + "_" +str(lr_cur) + "_t"+str(trial)
			file_string = main_string1 + "gpu" + main_string2 + bit_string + main_string3 + bit_string + main_string4 + bit_string + main_string5 + str(w_cur) + " -ab " + str(a_cur) + " -eb " + str(a_cur) + " -gb " + str(8) + " -lr " + str(lr_cur) + " -t " + str(trial)
			with open('jobscripts/snn_'+bit_string+'.script', 'w') as f:
				f.write(file_string)
			os.system("qsub "+'jobscripts/snn_'+bit_string+'.script')


	#gb sweep
	for w_cur in wb_sweep:
		for a_cur in gb_sweep:
			bit_string = str(w_cur) + "12" + str(a_cur) + "12" + "_" +str(lr_cur)+ "_t"+str(trial)
			file_string = main_string1 + "gpu" + main_string2 + bit_string + main_string3 + bit_string + main_string4 + bit_string + main_string5 + str(w_cur) + " -ab " + str(12) + " -eb " + str(12) + " -gb " + str(a_cur) + " -lr " + str(lr_cur)+ " -t " + str(trial)
			with open('jobscripts/snn_'+bit_string+'.script', 'w') as f:
				f.write(file_string)
			os.system("qsub "+'jobscripts/snn_'+bit_string+'.script')








