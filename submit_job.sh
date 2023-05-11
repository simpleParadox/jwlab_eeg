#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=1:00:00
# SBATCH --array=1-450
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000
# SBATCH --job-name=12m_tgm_eeg_to_w2v_inside_scaling_no_seed_second_run
#SBATCH --job-name=Feb_17_2023_12m_predicting_pre_w2v_window_100_200_store_embeddings
# SBATCH --output=out_files/%x-%j.out
#SBATCH --output=%x-%j.out

source ~/jwlab/bin/activate

# python classification/notebooks/cluster_analysis_perm_reg_overlap.py $SLURM_ARRAY_TASK_ID
python classification/notebooks/cluster_analysis_perm_reg_overlap.py -1

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=2 --mem-per-cpu=8000 --time=01:00:00