#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import setup_jwlab
import random
import sys
sys.path.insert(1, '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/classification/code')

from jwlab.constants import cleaned_data_filepath


from jwlab.cluster_analysis_perm import cluster_analysis_procedure
from jwlab.ml_prep_perm import prep_ml, slide_df, init, load_ml_data, get_bad_trials, map_participants,average_trials_and_participants
from jwlab.bad_trials import get_bad_trials, get_left_trial_each_word

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_validate, RepeatedKFold
from scipy import stats
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


# In[2]:


age_group = 12
useRandomizedLabel = False
averaging = "permutation_with_labels"
sliding_window_config = [-200, 1000, [100], 10]
downsample_num=1000


# In[ ]:


#set up variable inputs
matrix_build = 15 # change
split_itr = 4 # change
fold = 3
fold_fac= 1/fold

#set up table 
win_size= sliding_window_config[2][0]
step_size = sliding_window_config[3]
res = {}
temp_res ={}


for m in range(matrix_build):
    print("Matrix build: ", m+1)
    X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000)
    
    num_indices = len(X[0][0])
    #split say 15 times? 
    for s in range(split_itr):
        for f in range(fold):
            ind_key = []
            for i in range(0, num_indices-1, 1):
                ind_key.append(i)
            # random.shuffle(ind_key)
            fold_testsize = int(fold_fac*num_indices)
            if f == 0:
                test_indices = ind_key[:fold_testsize]
            elif f == 1:
                test_indices = ind_key[fold_testsize:(num_indices-fold_testsize)]
            elif f == 2:
                test_indices = ind_key[(num_indices-fold_testsize):]
            else:
                print('Code currently only current supports 3 folds')
            
            num_sliding_windows = len(X[0])

            df_test = []
            df_train = []

            X_train =[]
            y_train =[]
            X_test = [] 
            y_test = [] 

            for i in range(int(num_sliding_windows)):
                ## will need each window
                if 'level_0' in X[0][i].columns: 
                    X[0][i] = X[0][i].drop(columns = ['level_0'], axis = 1)
                X[0][i] = X[0][i].reset_index()   
                # #create new df with these indices and removing from orig
                df_test.append(X[0][i].iloc[test_indices])
                df_train.append(X[0][i].drop(X[0][i].index[test_indices]))
                assert(len(df_train[i]) + len(df_test[i]) == len(X[0][i]))
                df_test[i] = df_test[i].drop(columns=['index'], axis=1) 
                df_train[i] = df_train[i].drop(columns=['index'], axis=1)

                y_train.append(df_train[i].label.values)
                X_train.append(df_train[i].drop(columns = ['label', 'participant'], axis = 1))
                if 'level_0' in X_train[i].columns: 
                    X_train[i] = X_train[i].drop(columns = ['level_0'], axis = 1)
                y_test.append(df_test[i].label.values)
                X_test.append(df_test[i].drop(columns = ['label', 'participant'], axis = 1))
                if 'level_0' in X_test[i].columns: 
                    X_test[i] = X_test[i].drop(columns = ['level_0'], axis = 1)

                y_train[i][y_train[i] < 8] = 0
                y_train[i][y_train[i] >= 8] = 1
                y_test[i][y_test[i] < 8] = 0
                y_test[i][y_test[i] >= 8] = 1
                
                

            if len(num_win) > 1:
                print("Error: Function not supported for mutliple window lengths.")
            else: 
                for i in range(num_win[0]):
                     for j in range(num_win[0]):
# for debugging                    j = i
                        train_win = (i * step_size) - 200
                        test_win = (j * step_size) - 200

                        model = LinearSVC(C=1e-9, max_iter=100)
                        model.fit(X_train[i], y_train[i])
                        y_pred = model.predict(X_test[j])
                        testScore = accuracy_score(y_test[j], y_pred)

                        if (train_win,test_win) in temp_res:
                            temp_res[train_win, test_win].append(testScore)
                        else:
                            temp_res[train_win, test_win] = [testScore]


# In[ ]:


ind = np.arange(-200, 1000, step_size).tolist()
cols = np.arange(-200, 1000, step_size).tolist()

time_table = pd.DataFrame(index=ind, columns=cols)       
for i in range(num_win[0]):
        for j in range(num_win[0]):   
            train_win = (i * step_size) - 200
            test_win = (j * step_size) - 200
            avg = sum(temp_res[train_win, test_win]) / len(temp_res[train_win, test_win])
            time_table.loc[train_win, test_win]=avg
            
# print(time_table)


# In[ ]:


# #for debugging

# res={}
# for i in range(num_win[0]):   
#         train_win = (i * step_size) - 200
#         test_win = (i * step_size) - 200
#         res[train_win, test_win]= sum(temp_res[train_win, test_win])/len(temp_res[train_win, test_win])
        
#  res.values()           


# In[ ]:


time_table.to_csv('run 2 12m_100ms_10msSteps_verification3.csv')

'''
# #### Train and predict windows between age groups

# In[2]:


train_age_group = 9
test_age_group = 11
useRandomizedLabel = False
averaging = "permutation_with_labels"
sliding_window_config = [-200, 1000, [100], 50]
downsample_num=1000

#set up variable inputs
matrix_build = 20 # change
split_itr = 15 # change
fold = 3
fold_fac= 1/fold

#set up table 
win_size= sliding_window_config[2][0]
step_size = sliding_window_config[3]
res = {}
temp_res ={}


for m in range(matrix_build):
    X_train_all, y_train_all, good_trial_count_train, num_win_train = prep_ml(train_age_group, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000)
    X_test_all, y_test_all, good_trial_count_test, num_win_test = prep_ml(test_age_group, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000)


    num_indices_train = len(X_train_all[0][0])
    num_indices_test = len(X_test_all[0][0])
    
     
    for s in range(split_itr):
        for f in range(fold):
            
            ind_key_train = []
            for i in range(0, num_indices_train-1, 1):
                ind_key_train.append(i)
            random.shuffle(ind_key_train)
            fold_testsize_trainingSet = int(fold_fac*num_indices_train)
            if f == 0:
                test_indices_trainingSet = ind_key_train[:fold_testsize_trainingSet]
            elif f == 1:
                test_indices_trainingSet = ind_key_train[fold_testsize_trainingSet:(num_indices_train-fold_testsize_trainingSet)]
            elif f == 2:
                test_indices_trainingSet = ind_key_train[(num_indices_train-fold_testsize_trainingSet):]
            else:
                print('Code currently only current supports 3 folds')
            
            ind_key_test = []
            for i in range(0, num_indices_test-1, 1):
                ind_key_test.append(i)
            random.shuffle(ind_key_test)
            fold_testsize_testingSet = int(fold_fac*num_indices_test)
            if f == 0:
                test_indices_testingSet = ind_key_test[:fold_testsize_testingSet]
            elif f == 1:
                test_indices_testingSet = ind_key_test[fold_testsize_testingSet:(num_indices_test-fold_testsize_testingSet)]
            elif f == 2:
                test_indices_testingSet = ind_key_test[(num_indices_test-fold_testsize_testingSet):]
            else:
                print('Code currently only current supports 3 folds')
            
            
            
            
            num_sliding_windows = len(X_train_all[0])

            df_test = []
            df_train = []

            X_train =[]
            y_train =[]
            X_test = [] 
            y_test = [] 

            for i in range(int(num_sliding_windows)):
                ## will need each window
                if 'level_0' in X_train_all[0][i].columns: 
                    X_train_all[0][i] = X_train_all[0][i].drop(columns = ['level_0'], axis = 1)
                if 'level_0' in X_test_all[0][i].columns: 
                    X_test_all[0][i] = X_test_all[0][i].drop(columns = ['level_0'], axis = 1)
                X_train_all[0][i] = X_train_all[0][i].reset_index()   
                X_test_all[0][i] = X_test_all[0][i].reset_index() 
                # #create new df with these indices and removing from orig
                df_test.append(X_test_all[0][i].iloc[test_indices_testingSet])
                df_train.append(X_train_all[0][i].drop(X_train_all[0][i].index[test_indices_trainingSet]))
#                 assert(len(df_train[i]) + len(df_test[i]) == len(X_train_all[0][i]) + len(X_test_all[0][i]))
                assert(num_win_train == num_win_test)
                df_test[i] = df_test[i].drop(columns=['index'], axis=1) 
                df_train[i] = df_train[i].drop(columns=['index'], axis=1)

                y_train.append(df_train[i].label.values)
                X_train.append(df_train[i].drop(columns = ['label', 'participant'], axis = 1))
                if 'level_0' in X_train[i].columns: 
                    X_train[i] = X_train[i].drop(columns = ['level_0'], axis = 1)
                y_test.append(df_test[i].label.values)
                X_test.append(df_test[i].drop(columns = ['label', 'participant'], axis = 1))
                if 'level_0' in X_test[i].columns: 
                    X_test[i] = X_test[i].drop(columns = ['level_0'], axis = 1)

                y_train[i][y_train[i] < 8] = 0
                y_train[i][y_train[i] >= 8] = 1
                y_test[i][y_test[i] < 8] = 0
                y_test[i][y_test[i] >= 8] = 1
                
                

            if len(num_win_train) > 1:
                print("Error: Function not supported for mutliple window lengths.")
            else: 
                for i in range(num_win_train[0]):
                    for j in range(num_win_train[0]):
                        train_win = (i * step_size) - 200
                        test_win = (j * step_size) - 200

                        model = LinearSVC(C=1e-9, max_iter=1000)
                        model.fit(X_train[i], y_train[i])
                        y_pred = model.predict(X_test[j])
                        testScore = accuracy_score(y_test[j],y_pred)

                        if (train_win,test_win) in temp_res:
                            temp_res[train_win, test_win].append(testScore)
                        else:
                            temp_res[train_win, test_win] = [testScore]


# In[4]:


ind = np.arange(-200, 1000, step_size).tolist()
cols = np.arange(-200, 1000, step_size).tolist()

time_table = pd.DataFrame(index=ind, columns=cols)       
for i in range(num_win_train[0]):
        for j in range(num_win_train[0]):   
            train_win = (i * step_size) - 200
            test_win = (j * step_size) - 200
            avg = sum(temp_res[train_win, test_win])/len(temp_res[train_win, test_win])
            time_table.loc[train_win, test_win]=avg
            
print(time_table)


# In[5]:


time_table.to_csv('9mTrain_12mTest_100ms_50msSteps.csv')  


# #### Train and predict windows between age groups OLD

# In[2]:


# 9m training windows
age_group = 9
useRandomizedLabel = False
averaging = "permutation"
sliding_window_config = [-200, 1000, [100], 10]
downsample_num=1000

X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000)


# In[3]:


X_train = X
y_train = y


# In[6]:


# 12m testing windows
age_group = 12
useRandomizedLabel = False
averaging = "permutation"
sliding_window_config = [-200, 1000, [100], 100]
downsample_num=1000

X_test, y_test, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000)


# In[8]:


#time val

from sklearn.svm import SVC, LinearSVC
win_size= 100
res = {}
ind = np.arange(-200, 1000, win_size).tolist()
cols = np.arange(-200, 1000, win_size).tolist()
time_table = pd.DataFrame(index=ind, columns=cols)
if len(num_win) > 1:
    print("Error: Function not supported for mutliple window lengths.")
else: 
    for i in range(num_win[0]):
        for j in range(num_win[0]): 
            train_win = (i * win_size) - 200
            test_win = (j * win_size) - 200
            X_train_i = X_train[0][i]
            y_train_i = y_train[0][i]
            X_test_j = X_test[0][j]
            y_test_j = y_test[0][j]
            
            model = LinearSVC(C=1e-9, max_iter=1000)
            model.fit(X_train_i, y_train_i)
            y_pred = model.predict(X_test_j)
            testScore = accuracy_score(y_test_j,y_pred) 
            print(testScore)
            res[train_win, test_win] = testScore
            time_table.loc[train_win, test_win]=testScore
            
time_table.to_csv('9mSubsetTrain_12mTest_Timewin_Pred_100ms.csv')       


# In[ ]:


ind = np.arange(-200, 1000, win_size).tolist()
cols = np.arange(-200, 1000, win_size).tolist()
time_table = pd.DataFrame(index=ind, columns=cols)
time_table.loc[100, 400]=99
time_table


# In[ ]:





# In[8]:





# In[ ]:





# In[ ]:





# In[ ]:




'''