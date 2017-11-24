#!/bin/sh
#SBATCH --partition sgpu      # partition requested
#SBATCH -N 1      # nodes requested
#SBATCH -c 2      # cores requested
#SBATCH --output /projects/akar9135/1_first_run/test-ouput.txt
module load gcc
module load cudnn
module load cuda
module load python/3.5.1
source activate /projects/akar9135/sample/
module load python/3.5.1
python try.py
source deactivate
