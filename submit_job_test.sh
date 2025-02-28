#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=rrg-afyshe
#SBATCH --time=15:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=20000

source ~/base/bin/activate

# python regression/general_inspection.py
python classification/notebooks/Time_Window_Prediction_2.py