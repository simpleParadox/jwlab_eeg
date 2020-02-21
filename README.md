# jwlab_eeg
This is the repo for the eeg study in the Janet Werker lab

# current model scores

# SVMs on complete data sets
Notes: 

9m: n=6
svms (kernal=linear C=1e-9, after some amount of tuning)
- train avg, test avg: .4375 (training)
- train avg, test raw: .5002
- train raw, test raw: .2891 (training)
- train raw, test avg: .4375

12m: n=8
svms (kernal=linear C=1e-9, after some amount of tuning)
- train avg, test avg: .31 (training)
- train avg, test raw: .45
- train raw, test raw: .29 (training)
- train raw, test avg: .31

all kids together: 
svms (kernal=linear C=1e-9, after some amount of tuning)
- train avg, test avg: .4375 (training)
- train avg, test raw: .4702
- train raw, test raw: .2943 (training)
- train raw, test avg: .5625 

# directory structure

Code using the Matlab eeglab library runs in the data_cleaning folder, specifically the [all_cleaning.m](data_cleaning/all_cleaning.m) file.

After cleaning the data it is exported to a .csv file to be read and used by python files in the classification folder. The other thing we use matlab for is to get the labels from the data, thats in the [get_labels.m](data_cleaning/get_labels.m) file.

In the classification folder, the code is structured into .py helper function files, and the jupyter notebook files that actually run and evaluate the models. The most important files are [ml_prep.py](classification/ml_prep.py) file, which loads the data from the .csv and prepares it for classification, and the [ml_first_6_model_testing.ipynb](classification/ml_first_6_model_testing.ipynb) file, which runs the models on the data. The other jupyter notebook files do miscellaneous tasks like signifigance testing and sanity checks.

# Important cleaning steps :
- boundary removal
- filtering
- baseline removal
- avg referencing

# Project Structure: (library files used by MatLab under data cleanning are not listed here for simplicity)

- __jwlab\_eeg__
   - [README.md](README.md)
   - __Readings__
     - [1\-s2.0\-S0093934X10001811\-main.pdf](Readings/1-s2.0-S0093934X10001811-main.pdf)
     - [Catalyst\_05172019\_JCclean.docx](Readings/Catalyst_05172019_JCclean.docx)
     - [Sudre et al. \- 2012 \- Tracking Neural Coding of Perceptual and Semantic Features of Concrete Nouns.pdf](Readings/Sudre%20et%20al.%20-%202012%20-%20Tracking%20Neural%20Coding%20of%20Perceptual%20and%20Semantic%20Features%20of%20Concrete%20Nouns.pdf)
     - [fnsys\-02\-004.pdf](Readings/fnsys-02-004.pdf)
   - __classification__
     - __\_\_pycache\_\___
       - [bad\_trials.cpython\-37.pyc](classification/__pycache__/bad_trials.cpython-37.pyc)
       - [constants.cpython\-37.pyc](classification/__pycache__/constants.cpython-37.pyc)
       - [first\_participants\_map.cpython\-37.pyc](classification/__pycache__/first_participants_map.cpython-37.pyc)
       - [ml\_prep.cpython\-37.pyc](classification/__pycache__/ml_prep.cpython-37.pyc)
     - __code__
       - __jwlab__
         - [\_\_init\_\_.py](classification/code/jwlab/__init__.py)
         - __\_\_pycache\_\___
           - [\_\_init\_\_.cpython\-37.pyc](classification/code/jwlab/__pycache__/__init__.cpython-37.pyc)
           - [bad\_trials.cpython\-37.pyc](classification/code/jwlab/__pycache__/bad_trials.cpython-37.pyc)
           - [constants.cpython\-37.pyc](classification/code/jwlab/__pycache__/constants.cpython-37.pyc)
           - [first\_participants\_map.cpython\-37.pyc](classification/code/jwlab/__pycache__/first_participants_map.cpython-37.pyc)
           - [ml\_prep.cpython\-37.pyc](classification/code/jwlab/__pycache__/ml_prep.cpython-37.pyc)
         - [bad\_trials.py](classification/code/jwlab/bad_trials.py)
         - [constants.py](classification/code/jwlab/constants.py)
         - __dummy\_dataset\_classificaiton__
           - [computers\_classify.py](classification/code/jwlab/dummy_dataset_classificaiton/computers_classify.py)
         - [eval.py](classification/code/jwlab/eval.py)
         - [first\_participants\_map.py](classification/code/jwlab/first_participants_map.py)
         - [ml\_prep.py](classification/code/jwlab/ml_prep.py)
         - __run__
           - [\_\_init\_\_.py](classification/code/jwlab/run/__init__.py)
           - [basic\_sktime\_classifier.py](classification/code/jwlab/run/basic_sktime_classifier.py)
           - [computecanada\_constants.py](classification/code/jwlab/run/computecanada_constants.py)
           - [prep\_extracted\_df.py](classification/code/jwlab/run/prep_extracted_df.py)
           - [prep\_ml\_df.py](classification/code/jwlab/run/prep_ml_df.py)
           - [prep\_sktime\_df.py](classification/code/jwlab/run/prep_sktime_df.py)
           - [run\_setup.py](classification/code/jwlab/run/run_setup.py)
           - [sktime\_column\_ensemble.py](classification/code/jwlab/run/sktime_column_ensemble.py)
           - [test\_job.py](classification/code/jwlab/run/test_job.py)
           - [train\_eval\_svm\_noavg.py](classification/code/jwlab/run/train_eval_svm_noavg.py)
           - [train\_eval\_svm\_noavg\_avg.py](classification/code/jwlab/run/train_eval_svm_noavg_avg.py)
       - [setup.py](classification/code/setup.py)
     - __extracted\_features__
       - [ml\_prep\_tsfresh.py](classification/extracted_features/ml_prep_tsfresh.py)
     - __jobs__
       - [computer\_classify.sh](classification/jobs/computer_classify.sh)
       - [general\_requirements.txt](classification/jobs/general_requirements.txt)
       - [prep\_extracted\_df.sh](classification/jobs/prep_extracted_df.sh)
       - [prep\_ml\_df.sh](classification/jobs/prep_ml_df.sh)
       - [prep\_sktime\_df.sh](classification/jobs/prep_sktime_df.sh)
       - __saved\_outputs__
         - [sktime\_forest.out](classification/jobs/saved_outputs/sktime_forest.out)
         - [svm\_noavg.out](classification/jobs/saved_outputs/svm_noavg.out)
       - [sktime\_class.sh](classification/jobs/sktime_class.sh)
       - [train\_eval\_svm.sh](classification/jobs/train_eval_svm.sh)
       - [train\_eval\_svm\_noavg\_avg.sh](classification/jobs/train_eval_svm_noavg_avg.sh)
     - __notebooks__
       - __\_\_pycache\_\___
         - [setup\_jwlab.cpython\-37.pyc](classification/notebooks/__pycache__/setup_jwlab.cpython-37.pyc)
       - [analyze ml\_df.ipynb](classification/notebooks/analyze%20ml_df.ipynb)
       - [ml\_dtw\_first4\_testing.ipynb](classification/notebooks/ml_dtw_first4_testing.ipynb)
       - [ml\_extractedfeatures\_first4\_testing.ipynb](classification/notebooks/ml_extractedfeatures_first4_testing.ipynb)
       - [ml\_randomforest\_first4\_averagetrials.ipynb](classification/notebooks/ml_randomforest_first4_averagetrials.ipynb)
       - [ml\_svms\_first4\_noaveraging\_validate\_on\_averaged.ipynb](classification/notebooks/ml_svms_first4_noaveraging_validate_on_averaged.ipynb)
       - [ml\_svms\_first4\_noaveraging\_validate\_on\_averaged\_sigtest.ipynb](classification/notebooks/ml_svms_first4_noaveraging_validate_on_averaged_sigtest.ipynb)
       - [ml\_svms\_first4\_testing.ipynb](classification/notebooks/ml_svms_first4_testing.ipynb)
       - [sanity\_999.ipynb](classification/notebooks/sanity_999.ipynb)
       - [sanity\_real\_multiple\_participants.ipynb](classification/notebooks/sanity_real_multiple_participants.ipynb)
       - [setup\_jwlab.py](classification/notebooks/setup_jwlab.py)
       - [significance\_check.ipynb](classification/notebooks/significance_check.ipynb)
       - [temp\_test.ipynb](classification/notebooks/temp_test.ipynb)
       - [tsfresh\_extract.ipynb](classification/notebooks/tsfresh_extract.ipynb)
   - __data\_cleaning__
     - [all\_cleaning.m](data_cleaning/all_cleaning.m)
     - [clean\_bad\_channels.m](data_cleaning/clean_bad_channels.m)
     - [get\_labels.m](data_cleaning/get_labels.m)
     - [get\_trial\_cell\_obs.m](data_cleaning/get_trial_cell_obs.m)
        - __EEGLab__
        - __functions__
        - __plugins__
     - __jobs__
       - [cleaning.sl](data_cleaning/jobs/cleaning.sl)
   - __old\_scripts__
     - [BadChannelProcessing.m](old_scripts/BadChannelProcessing.m)
     - [Fil\_Epoch.m](old_scripts/Fil_Epoch.m)
     - [FilterChannelsToFrontal.m](old_scripts/FilterChannelsToFrontal.m)
     - [ML\_Matrix\_code\_multipleParticipants.m](old_scripts/ML_Matrix_code_multipleParticipants.m)
     - [ML\_Matrix\_code\_multipleParticipantsWORKS.m](old_scripts/ML_Matrix_code_multipleParticipantsWORKS.m)
     - [SVM\_Fit\_Iterator.m](old_scripts/SVM_Fit_Iterator.m)
     - [SanityCheck.m](old_scripts/SanityCheck.m)
   - __papers__
     - [1\-s2.0\-S0093934X10001811\-main.pdf](papers/1-s2.0-S0093934X10001811-main.pdf)
     - [Catalyst\_05172019\_JCclean.docx](papers/Catalyst_05172019_JCclean.docx)
     - [fnsys\-02\-004.pdf](papers/fnsys-02-004.pdf)

