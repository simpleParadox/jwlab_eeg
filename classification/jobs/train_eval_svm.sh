#!/bin/bash
#SBATCH --account=def-jwerker
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0:20:00

module load python/3.6
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r general_requirements.txt
python /$HOME/projects/def-jwerker/kjslakov/jwlab_eeg/classification/code/jwlab/run/train_eval_svm.py