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
                    
      

    pvalues, tvalues = t_test(results, num_win, num_folds)

    clusters = find_clusters(pvalues, tvalues)

    max_t_mass = get_max_t_mass(clusters, tvalues)
    
    ## REMOVE FOR NULL FUNCTION
    if len(sliding_window_config[2]) == 1:
        createGraph(results)
    else: 
        print("Graph function is not supported for multiple window sizes")

    return results

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
    pvalues = []
    tvalues = []
    for i in range(len(results)):
        for j in range(num_win[i]):
            # change the second argument below for comparison
            istat = stats.ttest_1samp(results[i][j], .5)
            pvalues += [istat.pvalue] if istat.statistic > 0 else [1]
            # removed just so that we can get the negative value from the pre window
            tvalues += [istat.statistic] if istat.statistic > 0 else [0]
    
    return pvalues, tvalues

# Finding contiguous time cluster
def find_clusters(pvalues, tvalues):
    #change to two tailed
    valid_window = [i for i,v in enumerate(pvalues) if v <= 0.025]
     ## REMOVE FOR NULL FUNCTION
    print("Valid windows are: {0}\n".format(valid_window))
    
    # Obtain clusters (3 or more consecutive meaningful time)
    clusters = [list(group) for group in mit.consecutive_groups(valid_window)]
    clusters = [group for group in clusters if len(group) >= 3]
    
    adj_clusters = []
    for c in clusters: 
        new_list = [((x*10)-200) for x in c]
        adj_clusters.append(new_list)
         
    ## REMOVE FOR NULL FUNCTION
    print("Clusters are: {0}\n".format(adj_clusters))
    return clusters

def get_max_t_mass(clusters, tvalues):
    t_mass = [0]
    for c in clusters:
        t_scores = 0
        for time in c:
            t_scores += tvalues[time]
        t_mass += [t_scores]
    
    max_t_mass = max(t_mass)
    
     ## REMOVE FOR NULL FUNCTION
    print("The max t mass is: {0}\n".format(max_t_mass))
    
    return max_t_mass