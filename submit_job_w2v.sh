#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000
#SBATCH --job-name=2025_Mar_1_12m_ph-12m-eeg_to_ph_updated_ridge_params_fixed_seed_checking_ridge_params_20_iters
#SBATCH --output=out_files/%x-%j.out

module load StdEnv/2023
module load scipy-stack
module load python/3.10
source ~/jwlab/bin/activate
echo 'Starting the job... with layer: '
echo $SLURM_ARRAY_TASK_ID
python classification/notebooks/cluster_analysis_perm_reg_overlap.py --seed=-1 \
       --layer=1 --age_group=12 --graph_file_name='2025_ph-12m-eeg_to_ph_fixed_seed' \
       --iterations=50 --fixed_seed --embedding_type='ph'

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=2 --mem-per-cpu=8000 --time=00:10:00 --mail-user=rsaha@ualberta.ca --mail-type=BEGIN