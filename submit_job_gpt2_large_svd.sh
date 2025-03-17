#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=05:40:00
#SBATCH --array=0
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000
#SBATCH --job-name=2025_Mar_13_9m_gpt2-large-9m-eeg_to_vectors_updated_ridge_params_fixed_seed_100_iters_svd_16
#SBATCH --output=gpt2_large_svd_16_9m_out_files/%x-%j.out

module load StdEnv/2023
module load scipy-stack
module load gcc opencv/4.10.0 # Must be in this position of the module loading process.
module load python/3.10
source ~/jwlab/bin/activate
echo 'Starting the job... with layer: '
echo $SLURM_ARRAY_TASK_ID

python classification/notebooks/cluster_analysis_perm_reg_overlap.py --seed=-1 \
       --layer=$SLURM_ARRAY_TASK_ID --age_group=9 --graph_file_name='2025_9m-gpt2-large_to_vectors_fixed_seed_svd_-16_layer' \
       --iterations=100 --fixed_seed --model_name='gpt2-large-svd-16' --svd_vectors

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=4 --mem-per-cpu=8000 --time=00:10:00 --mail-user=rsaha@ualberta.ca --mail-type=BEGIN