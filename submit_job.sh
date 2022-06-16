#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=rrg-afyshe
#SBATCH --time=3:00:00
# SBATCH --array=1-100
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000
#SBATCH --job-name=9m_cleaned2_combination_2_15-06-2022
#SBATCH --output=%x-%j.out

source ~/jwlab/bin/activate

python classification/notebooks/cluster_analysis_perm_reg_overlap.py

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=rrg-afyshe --cpus-per-task=1 --mem-per-cpu=8000 --time=1:00:00