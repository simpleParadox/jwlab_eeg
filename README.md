# jwlab_eeg
This is the repo for the eeg study in the Janet Werker lab

# current model scores

Notes: no bad trial/channel removal
update on Nov 28: removed four words that are unlikely known by participants, set those values to zero

svms (kernal=linear C=1e-6, after some amount of tuning)
- no averaging : 0.425
- average trials : 0.442
- average trials+participants : 0.423

random forest :
- no averaging : 0.56
- average trials : 0.55
- average trials+participants : 0.53

## current model scores on individual participants
svms
- no averaging : 107: 0.5625; 904: 0.5; 905: 0.4812; 906: 0.4562
- average trials : 107: 0.49375; 904: 0.50625; 905: 0.5482; 906: 0.5879
- average trials+participants : 107: 0.5258; 904: 0.49375; 905: 0.5; 906: 0.524

# directory structure

Code using the Matlab eeglab library runs in the data_cleaning folder, specifically the [all_cleaning.m](data_cleaning/all_cleaning.m) file.

After cleaning the data it is exported to a .csv file to be read and used by python files in the classification folder. The other thing we use matlab for is to get the labels from the data, thats in the [get_labels.m](data_cleaning/get_labels.m) file.

In the classification folder, the code is structured into .py helper function files, and the jupyter notebook files that actually run and evaluate the models. The most important files are [ml_prep.py](classification/ml_prep.py) file, which loads the data from the .csv and prepares it for classification, and the [ml_first_6_model_testing.ipynb](classification/ml_first_6_model_testing.ipynb) file, which runs the models on the data. The other jupyter notebook files do miscellaneous tasks like signifigance testing and sanity checks.

# Important cleaning steps :
- boundary removal
- filtering
- baseline removal
- avg referencing


