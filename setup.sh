module load StdEnv/2023
module load scipy-stack
module load python/3.10


# Create a virtual environment
virtualenv --no-download ~/jwlab
source ~/jwlab/bin/activate
pip install --no-index --upgrade pip


# Install the required packages
pip install scikit_learn
pip install tqdm
pip install more_itertools