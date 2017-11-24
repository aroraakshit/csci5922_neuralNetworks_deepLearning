#!/bin/sh
#SBATCH --partition=sgpu
#SBATCH -N 6     # nodes requested
#SBATCH -c 1      # cores requested
#SBATCH --output /projects/akar9135/3_first_run_lstm/test-ouput.txt
module load gcc
module load cudnn
module load cuda
module load python/3.5.1
source activate /projects/akar9135/sample/
module load python/3.5.1
python try_lstm.py
source deactivate
