#!/bin/bash -l
#SBATCH --job-name=all_cleaning
#SBATCH --account=def-jwerker
#SBATCH --time=0-1:30:00
#SBATCH --nodes=1      
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-user=karlslakov@gmail.com
#SBATCH --mail-type=ALL

module load matlab/2019b

srun matlab -nodisplay -singleCompThread -r "all_cleaning"