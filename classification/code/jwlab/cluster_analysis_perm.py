import pandas as pd
import numpy as np
from scipy import stats
import more_itertools as mit
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import RepeatedKFold
from jwlab.ml_prep_perm import prep_ml
from matplotlib import pyplot as plt

################################ Analysis procedure ################################

def cluster_analysis_procedure(age_group, useRandomizedLabel, averaging, sliding_window_config, cross_val_config):
    num_folds, cross_val_iterations, sampling_iterations = cross_val_config[0], cross_val_config[1], cross_val_config[2]
    
    results = {}

    for i in range(sampling_iterations):

        X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000)

        temp_results = cross_validaton(cross_val_iterations, num_win, num_folds, X, y)

        for i in range(len(temp_results)):
            if i not in results.keys():
                results[i] = {}
            for j in range(len(temp_results[i])):
                if j in results[i].keys(): 
                                    results[i][j] += temp_results[i][j]
                else:
                    results[i][j]= temp_results[i][j]
                    
      

    pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg = t_test(results, num_win, num_folds)

    clusters_pos, clusters_neg = find_clusters(pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg)

    max_t_mass = get_max_t_mass(clusters_pos, clusters_neg, tvalues_pos, tvalues_neg)
    
    ## REMOVE FOR NULL FUNCTION
    if len(sliding_window_config[2]) == 1:
        createGraph(results)
    else: 
        print("Graph function is not supported for multiple window sizes")

    return max_t_mass

def createGraph(results):
    scoreMean = []
    stdev = []
    for i in range(len(results)):
        for j in range(len(results[i])):

            scoreMean.append(round(np.mean(results[i][j]), 4))
            stdev.append(round(stats.sem(results[i][j]), 4))


    length_per_window_plt = 1200/ len(scoreMean)
    x_graph = np.arange(-200,1000,length_per_window_plt) 
    y_graph = scoreMean
    stdevplt = np.array(stdev)
    error = stdevplt
    plt.plot(x_graph, y_graph, 'k-')
    plt.fill_between(x_graph, y_graph-error, y_graph+error)
    plt.show()

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

                #model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
                model = LinearSVC(C=1e-9, max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                testScore = accuracy_score(y_test,y_pred) 
                
               
                if j in temp_results.keys(): 
                    temp_results[j] += [testScore]
                else:
                    temp_results[j] = [testScore]

            temp_scoreMean.append(round(np.mean(temp_results[j]), 2))
            #temp_stdev.append(round(np.std(temp_results[j]), 2))
            temp_stdev.append(round(stats.sem(temp_results[j]), 2))
        results.append(temp_results)
        scoreMean.append(temp_scoreMean)
        stdev.append(temp_stdev)
                
    for i in range(len(num_win)):
        for j in range(num_win[i]):
            assert len(results[i][j]) == num_iterations * num_folds
        
    return results


def t_test(results, num_win, num_folds):
    
    num_win= 120

    pvalues_pos = []
    pvalues_neg = []
    tvalues_pos = []
    tvalues_neg = []
    for i in range(len(results)):
        for j in range(num_win):
            # change the second argument below for comparison
            istat = stats.ttest_1samp(results[i][j], .5)
            pvalues_pos += [istat.pvalue] if istat.statistic > 0 else [1]
            pvalues_neg += [istat.pvalue] if istat.statistic < 0 else [1]
            # removed just so that we can get the negative value from the pre window
            tvalues_pos += [istat.statistic] if istat.statistic > 0 else [0]
            tvalues_neg += [istat.statistic] if istat.statistic < 0 else [0]
    return pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg


# Finding contiguous time cluster
def find_clusters(pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg):
    valid_window_pos = [i for i,v in enumerate(pvalues_pos) if v <= 0.05] 
    valid_window_neg = [i for i,v in enumerate(pvalues_neg) if v <= 0.05] 
    ## REMOVE FOR NULL FUNCTION
    print("Valid positive windows are: {0}\n".format(valid_window_pos))
    print("Valid negative windows are: {0}\n".format(valid_window_neg))

    # Obtain clusters (3 or more consecutive meaningful time)
    clusters_pos = [list(group) for group in mit.consecutive_groups(valid_window_pos)]
    clusters_pos = [group for group in clusters_pos if len(group) >= 3]

    clusters_neg = [list(group) for group in mit.consecutive_groups(valid_window_neg)]
    clusters_neg = [group for group in clusters_neg if len(group) >= 3]

    adj_clusters_pos = []
    for c in clusters_pos: 
        new_list = [((x*10)-200) for x in c]
        adj_clusters_pos.append(new_list)


    adj_clusters_neg = []
    for c in clusters_neg: 
        new_list = [((x*10)-200) for x in c]
        adj_clusters_neg.append(new_list)
         
    ## REMOVE FOR NULL FUNCTION
    print("Positive clusters are: {0}\n".format(adj_clusters_pos))
    print("Negative clusters are: {0}\n".format(adj_clusters_neg))
    return clusters_pos, clusters_neg

def get_max_t_mass(clusters_pos, clusters_neg, tvalues_pos, tvalues_neg):
    t_mass_pos = [0]
    for c in clusters_pos:
        t_scores_pos = 0
        for time in c:
            t_scores_pos += tvalues_pos[time]
        t_mass_pos += [t_scores_pos]

    max_t_mass_pos = max(t_mass_pos)
    print(max_t_mass_pos)

    t_mass_neg = [0]
    for c in clusters_neg:
        t_scores_neg = 0
        for time in c:
            t_scores_neg += tvalues_neg[time]
        t_mass_neg += [t_scores_neg]

    max_t_mass_neg = min(t_mass_neg)
    print(max_t_mass_neg)

    max_abs_tmass = max(max_t_mass_pos, abs(max_t_mass_neg))
    print(max_abs_tmass)

    
     ## REMOVE FOR NULL FUNCTION
    print("The max positive t mass is: {0}\n".format(max_t_mass_pos))
    print("The max negative t mass is: {0}\n".format(max_t_mass_neg))
    print("The max absolute t mass is: {0}\n".format(max_abs_tmass))
    
    return max_abs_tmass
