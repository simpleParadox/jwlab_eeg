#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
# SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=04:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000
#SBATCH --job-name=2025_Mar_16_9m_ph-9m-eeg_to_vectors_updated_ridge_params_fixed_seed_50_iters_range_150_200
#SBATCH --output=out_files/%x-%j.out

module load StdEnv/2023
module load scipy-stack
module load python/3.10
source ~/jwlab/bin/activate
echo 'Starting the job... with layer: '
echo $SLURM_ARRAY_TASK_ID
python classification/notebooks/cluster_analysis_perm_reg_overlap.py --seed=-1 \
       --layer=1 --age_group=9 --graph_file_name='2025_mar_16_ph-9m-eeg_to_vectors_fixed_seed_50_iters_range_150_200' \
       --embedding_type='ph' --iteration_range 150 200 --fixed_seed


# NOTE: You must use the --fixed_seed flag to use the fixed seed variant.
# --model_name=None # this is for w2v embeddings.
# --model_name='gpt2-large'
##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=2 --mem-per-cpu=8000 --time=00:10:00 --mail-user=rsaha@ualberta.ca --mail-type=BEGIN