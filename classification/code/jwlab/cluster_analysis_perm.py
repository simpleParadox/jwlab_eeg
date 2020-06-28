import pandas as pd
import numpy as np
from scipy import stats
import more_itertools as mit
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import RepeatedKFold
from jwlab.ml_prep_perm import prep_ml

################################ Analysis procedure ################################

def cluster_analysis_procedure(age_group, useRandomizedLabel, averaging, sliding_window_config, cross_val_config):
    num_folds, num_iterations = cross_val_config[0], cross_val_config[1]

    X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000)

    results = cross_validaton(num_iterations, num_win, num_folds, X, y)

    pvalues = t_test(results, num_iterations, num_win, num_folds)

    clusters = find_clusters(pvalues)

    max_t_mass = get_max_t_mass(clusters, pvalues)

    return max_t_mass

def cross_validaton(num_iterations, num_win, num_folds, X, y):
    results = []
    rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_iterations)
    scoreMean = []
    stdev = []
    
    for i in range(len(num_win)):
        temp_scoreMean = []
        temp_stdev = []
        temp_results = {}
        for j in range(num_win[i]):
            X_temp = X[i][j]
            y_temp = y[i][j]

            for train_index, test_index in rkf.split(X_temp):
                X_train, X_test = X_temp[train_index], X_temp[test_index]
                y_train, y_test = y_temp[train_index], y_temp[test_index]

                #model = SVC(kernel = 'rbf')
                model = LinearSVC(C=1e-9, max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                testScore = accuracy_score(y_test,y_pred)

                if j in temp_results.keys(): 
                    temp_results[j] += [testScore]
                else:
                    temp_results[j] = [testScore]

            temp_scoreMean.append(round(np.mean(temp_results[j]), 2))
            temp_stdev.append(round(np.std(temp_results[j]), 2))
        results.append(temp_results)
        scoreMean.append(temp_scoreMean)
        stdev.append(temp_stdev)
                
    for i in range(len(num_win)):
        for j in range(num_win[i]):
            assert len(results[i][j]) == num_iterations * num_folds
        
    print("mean: {0}".format(scoreMean))
    print("stdev: {0}".format(stdev))
    
    return results

def t_test(results, num_iterations, num_win, num_folds):
    pvalues = []
    for i in range(len(results)):
        for j in range(num_win[i]):
            istat = stats.ttest_1samp(results[i][j], .5)
            pvalues += [istat.pvalue] if istat.statistic > 0 else [1]
    
    return pvalues

# Finding contiguous time cluster
def find_clusters(pvalues):
    valid_window = [i for i,v in enumerate(pvalues) if v <= 0.025]
    print("Valid windows are: {0}\n".format(valid_window))
    
    # Obtain clusters (3 or more consecutive meaningful time)
    clusters = [list(group) for group in mit.consecutive_groups(valid_window)]
    clusters = [group for group in clusters if len(group) >= 3]
    print("Clusters are: {0}\n".format(clusters))
    
    return clusters

def get_max_t_mass(clusters, pvalues):
    t_mass = []
    for c in clusters:
        t_scores = 0
        for time in c:
            t_scores += pvalues[time]
        t_mass += [t_scores]
    
    max_t_mass = max(t_mass)
    print("The max t mass is: {0}\n".format(max_t_mass))
    
    return max_t_mass