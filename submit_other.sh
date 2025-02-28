#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=rrg-afyshe
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=10000
#SBATCH --job-name=9m_res_tgm_perm_1
#SBATCH --output=%x-%j.out

source ~/base/bin/activate

python classification/notebooks/cluster_analysis_perm_reg_overlap_parallel.py

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50