#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
# SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=00:40:00
#SBATCH --array=0-59
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --job-name=2025_Mar_23_9m_w2v_group_channel_50_iters_fixed_seed_range_50_100_window_440_660
#SBATCH --output=/home/rsaha/scratch/jwlab_eeg/group_channel_out_files/%x-%j.out


module load StdEnv/2023
module load scipy-stack
module load python/3.10
source ~/jwlab/bin/activate
echo 'Starting the job... with layer: '
echo $SLURM_ARRAY_TASK_ID
python classification/notebooks/cluster_analysis_perm_reg_overlap.py --seed=-1 \
       --layer=1 --age_group=9 --graph_file_name='2025_mar_23_w2v-9m-fixed_seed_group_channel' \
       --embedding_type='w2v' --iteration_range 50 100 --fixed_seed \
       --ch_group --group_num=$SLURM_ARRAY_TASK_ID --window_range 440 660 \
       --window_length=220 --wandb_mode='offline' --store_dir='/home/rsaha/scratch/jwlab_eeg'

# Use the following command for testing.
# python classification/notebooks/cluster_analysis_perm_reg_overlap.py --seed=-1 \
#        --layer=1 --age_group=12 --graph_file_name='2025_mar_22_w2v-12m-fixed_seed_group_channel_50_iters_seed_50_53' \
#        --embedding_type='w2v' --iteration_range 50 53 --fixed_seed \
#        --ch_group --group_num=0 --window_range 400 700 \
#        --window_length=300 --wandb_mode='offline' 
##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=4 --mem-per-cpu=8000 --time=00:10:00 --mail-user=rsaha@ualberta.ca --mail-type=BEGIN