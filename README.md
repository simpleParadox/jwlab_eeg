# jwlab_eeg
This is the repo for the eeg study in the Janet Werker lab

# current model scores
Update on 2020 Feb 29: now the participants are separated into a training_set (80%) and a test_set(20%) randomly.

Update on 2020 Feb 22: bad trials and channels are moved in the preparation step now.
~~Notes: no bad trial/channel removal~~

## svms (kernal=linear C=1e-6, after some amount of tuning)
### All participants
- raw: 0.47

- average_trials: 0.51

- fully_averaged: 0.62

### All 13-month participants:
- raw: 0.39

- average_trials: 0.43

- fully_averaged: 0.47

### All 9-month participants:
- raw: 0.43

- average_trials: 0.57

- fully_averaged: 0.47

## random forest :
- no averaging : 0.49
- average trials : 0.489
- average trials+participants : 0.585

# directory structure

Code using the Matlab eeglab library runs in the data_cleaning folder, specifically the [all_cleaning.m](data_cleaning/all_cleaning.m) file.

After cleaning the data it is exported to a .csv file to be read and used by python files in the classification folder. The other thing we use matlab for is to get the labels from the data, thats in the [get_labels.m](data_cleaning/get_labels.m) file.

In the classification folder, the code is structured into .py helper function files, and the jupyter notebook files that actually run and evaluate the models. The most important files are [ml_prep.py](classification/ml_prep.py) file, which loads the data from the .csv and prepares it for classification, and the [ml_first_6_model_testing.ipynb](classification/ml_first_6_model_testing.ipynb) file, which runs the models on the data. The other jupyter notebook files do miscellaneous tasks like signifigance testing and sanity checks.

# Important cleaning steps :
- boundary removal
- filtering
- baseline removal
- avg referencing
