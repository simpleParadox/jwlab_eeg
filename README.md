# jwlab_eeg
This is the repo for the eeg study in the Janet Werker lab

# current model scores

Notes: no bad trial/channel removal

9m: n=6
svms (kernal=linear C=1e-9, after some amount of tuning)
- train avg, test avg: 0.458
- train avg, test raw: 0.4805
- train raw, test raw: 0.28
- train raw, test avg: 0.458

12m: 
svms (kernal=linear C=1e-9, after some amount of tuning)
- train avg, test avg:0.565
- train avg, test raw:0.539
- train raw, test raw:0.2742
- train raw, test avg:0.565

all kids together: 
svms (kernal=linear C=1e-9, after some amount of tuning)
- train avg, test avg:0.5555
- train avg, test raw:0.5345
- train raw, test raw:0.2968
- train raw, test avg:0.55555


# directory structure

Code using the Matlab eeglab library runs in the data_cleaning folder, specifically the [all_cleaning.m](data_cleaning/all_cleaning.m) file.

After cleaning the data it is exported to a .csv file to be read and used by python files in the classification folder. The other thing we use matlab for is to get the labels from the data, thats in the [get_labels.m](data_cleaning/get_labels.m) file.

In the classification folder, the code is structured into .py helper function files, and the jupyter notebook files that actually run and evaluate the models. The most important files are [ml_prep.py](classification/ml_prep.py) file, which loads the data from the .csv and prepares it for classification, and the [ml_first_6_model_testing.ipynb](classification/ml_first_6_model_testing.ipynb) file, which runs the models on the data. The other jupyter notebook files do miscellaneous tasks like signifigance testing and sanity checks.

# Important cleaning steps :
- boundary removal
- filtering
- baseline removal
- avg referencing


