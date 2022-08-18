#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=16:00:00
# SBATCH --array=1-450
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000
# SBATCH --job-name=12m_tgm_bad_remove_cleaned2_w2v_mod_2v2
#SBATCH --job-name=12m_animacy_cleaned2_20_iters
# SBATCH --output=out_files/%x-%j.out
#SBATCH --output=%x-%j.out

source ~/jwlab/bin/activate

# python classification/notebooks/cluster_analysis_perm_reg_overlap.py $SLURM_ARRAY_TASK_ID
python classification/notebooks/cluster_analysis_perm_reg_overlap.py -1

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=1 --mem-per-cpu=4000 --time=00:10:00