#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=2:30:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-100
#SBATCH --mem-per-cpu=4000
#SBATCH --job-name=Sep_1_2023_9m_trial_dist_from_eeg_0_300ms_window
#SBATCH --output=perm_out_files/%x-%j.out

source ~/jwlab/bin/activate

python classification/notebooks/cluster_analysis_perm_reg_overlap.py $SLURM_ARRAY_TASK_ID

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=2 --mem-per-cpu=8000 --time=01:00:00