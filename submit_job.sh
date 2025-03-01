#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=5:30:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8000
#SBATCH --job-name=2025_Feb_28_9m_gpt2-large-9m-eeg_to_vectors
#SBATCH --output=out_files/%x-%j.out

source ~/jwlab/bin/activate

python classification/notebooks/cluster_analysis_perm_reg_overlap.py --seed=-1 \
       --model_name='gpt2-large' --layer=1 --age_group=9 --graph_file_name='gpt2-large-9m-eeg_to_vectors'

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=2 --mem-per-cpu=8000 --time=00:10:00 --mail-user=rsaha@ualberta.ca --mail-type=BEGIN