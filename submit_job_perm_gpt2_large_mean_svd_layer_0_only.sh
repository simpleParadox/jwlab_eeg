#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=4:30:00
#SBATCH --array=1-100
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000
#SBATCH --job-name=2025_Mar_17_perm_12m_gpt2-large-mean-12m-eeg_to_vectors_updated_ridge_params_fixed_seed_50_iters_range_50-100
#SBATCH --output=gpt2_large-mean_svd_16_perm_12m_out_files/%x-%j.out

module load StdEnv/2023
module load scipy-stack
module load gcc opencv/4.10.0 # Must be in this position of the module loading process.
module load python/3.10
source ~/jwlab/bin/activate
echo 'Running permutation job for layer 0 only: '

python classification/notebooks/cluster_analysis_perm_reg_overlap.py --seed=-1 \
       --layer=0 --age_group=12 --graph_file_name='2025_perm_12m-gpt2-large-mean_to_vectors_fixed_seed_svd_-16_layer' \
       --model_name='gpt2-large-mean-svd-16' --iteration_range 50 100 --svd_vectors --use_randomized_label

# NOTE: iteration_range really doesn't matter here. It's just a placeholder for the sake of consistency with other scripts.
# This is because the --fixed_seed flag is not provided. Furthermore, the --use_randomized_label flag is provided which shuffles the 
# labels for each iteration. So, the iteration range is not used in this case.

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=4 --mem-per-cpu=8000 --time=00:10:00 --mail-user=rsaha@ualberta.ca --mail-type=BEGIN