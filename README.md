# jwlab_eeg
This is the repo for the eeg study in the Janet Werker lab

# current model scores

Notes: no bad trial/channel removal

svms (kernal=linear C=1e-6, after some amount of tuning)
no averaging : 0.4408
average trials : 0.4923
average trials+participants : 0.685

random forest :
no averaging : 0.49
average trials : 0.489
average trials+participants : 0.585

# directory structure

Code using the Matlab eeglab library runs in the data_cleaning folder, specifically the [all_cleaning.m](data_cleaning/all_cleaning.m) file.

After cleaning the data it is exported to a .csv file to be read and used by python files in the classification folder. The other thing we use matlab for is to get the labels from the data, thats in the [get_labels.m](data_cleaning/get_labels.m) file.

In the classification folder, the code is structured into .py helper function files, and the jupyter notebook files that actually run and evaluate the models. The most important files are [ml_prep.py](classification/ml_prep.py) file, which loads the data from the .csv and prepares it for classification, and the [ml_first_6_model_testing.ipynb](classification/ml_first_6_model_testing.ipynb) file, which runs the models on the data. The other jupyter notebook files do miscellaneous tasks like signifigance testing and sanity checks.

# Important cleaning steps :
- boundary removal
- filtering
- baseline removal
- avg referencing


