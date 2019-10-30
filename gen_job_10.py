import numpy as np

main_string1 = """#!/bin/csh

#$ -M cschaef6@nd.edu	 # Email address for job notification
#$ -m abe
#$ -q """

main_string2 = """# Specify queue (use ‘debug’ for development) // gpu@qa-2080ti-001.crc.nd.edu , qa-2080ti-002.crc.nd.edu , ta-titanv-001.crc.nd.edu, qa-titanx-001.crc.nd.edu, qa-1080ti-001.crc.nd.edu, da-1080-001.crc.nd.edu
#$ -l gpu_card=1 
#$ -N snn_01_wage_"""
main_string3 = """     # Specify job name
#$ -o ./logs/output_snn_nd01_"""
main_string3 = """_wage.txt
#$ -e ./logs/error_snn_nd01_"""
main_string3 = """_wage.txt

module load python       # Required modules
setenv OMP_NUM_THREADS $NSLOTS

python spytorch_precise_10.py """
main_string3 = """ -wb 2 -ab 6 -eb 6 -gb 8"""

wb_sweep = [2,3,4,5]
ab_sweep = [6,7,8,10,12]
gb_sweep = [8,9,10,11,12]


#ab sweep
for wb_cur in wb_sweep:
	for ab_cur in ab_sweep:
		file_string = 

#gb sweep

