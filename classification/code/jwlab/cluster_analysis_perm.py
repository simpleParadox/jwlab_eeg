import pandas as pd
import numpy as np
import random
from scipy import stats
import more_itertools as mit
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.metrics import accuracy_score

sys.path.insert(1, '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/classification/code')  ## For loading the following files.

from jwlab.ml_prep_perm import prep_ml, prep_matrices_avg
from matplotlib import pyplot as plt

sys.path.insert(1, '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg')
from regression.functions import get_w2v_embeds_from_dict, two_vs_two, extended_2v2_phonemes, extended_2v2_perm, \
    get_phoneme_onehots, get_phoneme_classes, get_sim_agg_first_embeds, remove_data, get_sim_agg_second_embeds, extended_2v2
from sklearn.linear_model import Ridge


def cluster_analysis_procedure(age_group, useRandomizedLabel, averaging, sliding_window_config, cross_val_config):
    print("Cluster analysis procedure")
    num_folds, cross_val_iterations, sampling_iterations = cross_val_config[0], cross_val_config[1], cross_val_config[2]

    results = {}
    tgm_results = []
    flag = 0

    for i in range(sampling_iterations):
        print("Sampling iteration: ", i)
        if averaging == "permutation":
            X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config,
                                                      downsample_num=1000)

            # X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel)  # This is a new statement.

            temp_results = cross_validaton(cross_val_iterations, num_win, num_folds, X, y)

            # temp_results = cross_validaton_averaging(X_train, X_test, y_train, y_test, useRandomizedLabel)

        elif averaging == "average_trials_and_participants":
            flag = 0

            X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, "no_average_labels",
                                                      sliding_window_config, downsample_num=1000)

            X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel)
            temp_results, temp_diag_tgm = cross_validaton_nested(X_train, y_train, X_test, y_test)
            tgm_results.append(temp_diag_tgm)

            if sampling_iterations == 0:
                print("Warning: This does not do fold validation")
        elif averaging == "tgm":
            start_index = 0
            end_index = 116
            flag = 1
            X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, "no_average_labels",
                                                      sliding_window_config, downsample_num=1000)

            X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel)
            temp_results = cross_validaton_tgm(X_train, y_train, X_test, y_test, start_index, end_index)
            tgm_results.append(temp_results)  # The temp_results is expected to be a square matrix.
        elif averaging == 'across':
            # Training on one group and testing on another group.
            age_group_1 = 9
            age_group_2 = 12
            X_1, y_1, good_trial_count_1, num_win_1 = prep_ml(age_group_1, useRandomizedLabel, "no_average_labels",
                                                      sliding_window_config, downsample_num=1000)

            X_2, y_2, good_trial_count_2, num_win_2 = prep_ml(age_group_2, useRandomizedLabel, "no_average_labels",
                                                              sliding_window_config, downsample_num=1000)

            X_train_1, X_test_1, y_train_1, y_test_1 = prep_matrices_avg(X_1, age_group_1, useRandomizedLabel, True,0)

            # Now the other group

            X_train_2, X_test_2, y_train_2, y_test_2 = prep_matrices_avg(X_2, age_group_2, useRandomizedLabel, False)
            temp_results, temp_diag_tgm = cross_validaton_nested(X_train_1, y_train_1, X_test_2, y_test_2)
            tgm_results.append(temp_results)





        else:
            print("Warning: This will only use the requested averaging matrix to perform a cross val")
            X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config,
                                                      downsample_num=1000)

            temp_results = cross_validaton(cross_val_iterations, num_win, num_folds, X, y)

    if flag == 0:
        for i in range(len(temp_results)):
            if i not in results.keys():
                results[i] = {}
            for j in range(len(temp_results[i])):
                if j in results[i].keys():
                    results[i][j] += temp_results[i][j]
                else:
                    results[i][j] = temp_results[i][j]
    else:
        # Averaging was of type 'tgm'.
        # Calculate average of all the matrices.
        final_tgm = np.mean(tgm_results, axis=0)
        # Save the tgm in a csv file.
        step_size = sliding_window_config[3]
        ind = np.arange(-200, 1000, step_size).tolist()
        cols = np.arange(-200, 1000, step_size).tolist()
        df = pd.DataFrame(data=final_tgm, index=ind, columns=cols)
        df.to_csv(f"11-11-2020 {age_group}m tgm 50ms 10ms {start_index}-{end_index} global scaled permuted 50 iters.csv")

    if flag == 0:

        pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg = t_test(results, num_win, num_folds)

        clusters_pos, clusters_neg = find_clusters(pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg)

        max_t_mass = get_max_t_mass(clusters_pos, clusters_neg, tvalues_pos, tvalues_neg)

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

    length_per_window_plt = 1200 / len(scoreMean)
    x_graph = np.arange(-200, 1000, length_per_window_plt)
    y_graph = scoreMean
    stdevplt = np.array(stdev)
    error = stdevplt
    plt.clf()
    plt.plot(x_graph, y_graph, 'k-')
    plt.ylim(0.3, 0.80)
    plt.fill_between(x_graph, y_graph - error, y_graph + error)
    plt.savefig("09-11-2020 avg_trials_and_ps nested cv 9m 50ms 10ms scaled one-hot")


def shuffle_labels(y_train, y_test):
    train_len = len(y_train)
    test_len = len(y_test)

    all_labels = np.concatenate((y_train, y_test), axis=0)
    np.random.shuffle(all_labels)
    random.shuffle(all_labels)

    y_train_shuffled = all_labels[:train_len]
    y_test_shuffled = all_labels[test_len:]

    return y_train_shuffled, y_test_shuffled


def average_data_on_labels(X, y):
    """
    Note: X contains a column called labels. Will use this average those columns together.
    X and y are correctly matched with the EEG data and the labels.
    """
    x_array = []
    y_array = []
    y_set = set(y)
    y_set_list = list(y_set)
    for label in y_set_list:
        x_array.append(X[X.label == label].iloc[:, :-2].values.mean(axis=0))
        y_array.append(label)
    return np.array(x_array), y_array


def cross_validaton(num_iterations, num_win, num_folds, X, y):
    results = []
    # rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_iterations)
    rkf = ShuffleSplit(n_splits=num_folds, test_size=.20)
    scoreMean = []
    stdev = []

    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

    for i in range(len(num_win)):
        temp_scoreMean = []
        temp_stdev = []
        temp_results = {}
        for j in range(num_win[i]):
            X_temp = X[i][j]
            y_temp = y[i][j]

            for train_index, test_index in rkf.split(X_temp):
                X_train, X_test = X_temp.iloc[train_index, :], X_temp.iloc[test_index, :]
                y_train, y_test = y_temp[train_index], y_temp[test_index]

                y_train_labels = get_w2v_embeds_from_dict(y_train)
                y_test_labels = get_w2v_embeds_from_dict(y_test)

                X_train = X_train.drop(labels=['label', 'participant'], axis=1)  # Reshape X_train by dropping labels.
                # Now average the test data here -> X_test and y_test.

                X_test, y_test = average_data_on_labels(X_test, y_test)

                # model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
                model = Ridge()
                clf = GridSearchCV(model, ridge_params, scoring='neg_mean_squared_error', n_jobs=6, cv=5)
                clf.fit(X_train, y_train_labels)

                y_pred = clf.predict(X_test)
                points, total_points, testScore, gcf, grid = extended_2v2(y_test_labels, y_pred)

                if j in temp_results.keys():
                    temp_results[j] += [testScore]
                else:
                    temp_results[j] = [testScore]

            temp_scoreMean.append(round(np.mean(temp_results[j]), 2))
            # temp_stdev.append(round(np.std(temp_results[j]), 2))
            temp_stdev.append(round(stats.sem(temp_results[j]), 2))
        results.append(temp_results)
        scoreMean.append(temp_scoreMean)
        stdev.append(temp_stdev)

    for i in range(len(num_win)):
        for j in range(num_win[i]):
            assert len(results[i][j]) == 1 * num_folds

    return results


def cross_validaton_averaging(X_train, X_test, y_train, y_test, useRandomizedLabel):
    results = []

    for i in range(len(X_train)):
        temp_results = {}
        for j in range(len(X_train[i])):

            # model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
            # model = LinearSVC(C=1e-9, max_iter=1000)
            model = Ridge()
            y_train_labels = get_w2v_embeds_from_dict(y_train[i][j])
            y_test_labels = get_w2v_embeds_from_dict(y_test[i][j])

            # if useRandomizedLabel:
            #     print("Randomized Labels")
            #     y_train_labels, y_test_labels = shuffle_labels(y_train_labels, y_test_labels)

            model.fit(X_train[i][j], y_train_labels)
            y_pred = model.predict(X_test[i][j])
            points, total_points, testScore, gcf, grid = extended_2v2(y_test_labels, y_pred)

            if j in temp_results.keys():
                temp_results[j] += [testScore]
            else:
                temp_results[j] = [testScore]

        results.append(temp_results)

    return results


def cross_validaton_nested(X_train, y_train, X_test, y_test):
    results = []
    tgm_matrix_temp = np.zeros((120, 120))
    # scoring = 'accuracy'
    scoring = 'neg_mean_squared_error'

    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

    for i in range(len(X_train)):
        temp_results = {}
        for j in range(len(X_train[i])):

            # this is for predicting the second phoneme only (sim_agg.csv).
            # First remove the data for which the second phoneme is not present.
            # NOTE: The remove data function is not being used because phoneme alternatives are now being used.
            # X_train[i][j], y_train[i][j] = remove_data(X_train[i][j], y_train[i][j])
            # X_test[i][j], y_test[i][j] = remove_data(X_test[i][j], y_test[i][j])

            # model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
            # model = LinearSVC(C=1e-9, max_iter=1000)

            # y_train_labels = get_w2v_embeds_from_dict(y_train[i][j])
            # y_test_labels = get_w2v_embeds_from_dict(y_test[i][j])

            # One-hot vectors here.
            # y_train_labels = get_phoneme_classes(y_train[i][j])
            # y_test_labels = get_phoneme_classes(y_test[i][j])

            # Get first sim_agg embeddings here.
            y_train_labels = get_sim_agg_first_embeds(y_train[i][j])
            y_test_labels = get_sim_agg_first_embeds(y_test[i][j])
            which_phoneme = 1

            # Get second sim_agg embeddings here
            # y_train_labels = get_sim_agg_second_embeds(y_train[i][j])
            # y_test_labels = get_sim_agg_second_embeds(y_test[i][j])
            # which_phoneme = 2


            # model = LogisticRegression(multi_class='multinomial')
            model = Ridge()

            clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=12, cv=5)

            clf.fit(X_train[i][j], y_train_labels)


            y_pred = clf.predict(X_test[i][j])
            # points, total_points, testScore, gcf, grid = two_vs_two(y_test_labels, y_pred)
            points, total_points, testScore, gcf, grid = extended_2v2_phonemes(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)

            # testScore = accuracy_score(y_test_labels, y_pred)

            tgm_matrix_temp[j, j] = testScore

            if j in temp_results.keys():
                temp_results[j] += [testScore]
            else:
                temp_results[j] = [testScore]

        results.append(temp_results)

    return results, tgm_matrix_temp


def cross_validaton_tgm(X_train, y_train, X_test, y_test, start, end):
    # results = []

    if end == 116:
        end = len(X_train[0])
    tgm_matrix_temp = np.zeros((120, 120))

    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

    for i in range(len(X_train)):
        temp_results = {}
        for j in range(start, end):

            y_train_labels = get_w2v_embeds_from_dict(y_train[i][j])
            model = Ridge()
            clf = GridSearchCV(model, ridge_params, scoring='neg_mean_squared_error', n_jobs=18, cv=5)
            clf.fit(X_train[i][j], y_train_labels)

            for k in range(len(X_train[i])):
                y_test_labels = get_w2v_embeds_from_dict(y_test[i][k])

                # scaler = StandardScaler()
                # X_train_scaled = scaler.fit_transform(X_train[i][j])
                # X_test_scaled = scaler.transform(X_test[i][j])
                # clf.fit(X_train_scaled, y_train_labels)

                # y_pred = clf.predict(X_test_scaled)
                y_pred = clf.predict(X_test[i][k])
                points, total_points, testScore, gcf, grid = two_vs_two(y_test_labels, y_pred)

                tgm_matrix_temp[j, k] = testScore
            #     if k in temp_results.keys():
            #         temp_results[j] += [testScore]
            #     else:
            #         temp_results[j] = [testScore]
            # if j in temp_results.keys():
            #     temp_results[j] += [testScore]
            # else:
            #     temp_results[j] = [testScore]
            #
            # results.append(temp_results)

    return tgm_matrix_temp


def t_test(results, num_win, num_folds):
    pvalues_pos = []
    pvalues_neg = []
    tvalues_pos = []
    tvalues_neg = []
    for i in range(len(results)):
        for j in range(num_win[i]):
            # change the second argument below for comparison
            istat = stats.ttest_1samp(results[i][j], .5)
            pvalues_pos += [istat.pvalue] if istat.statistic > 0 else [1]
            pvalues_neg += [istat.pvalue] if istat.statistic < 0 else [1]
            # removed just so that we can get the negative value from the pre window
            tvalues_pos += [istat.statistic] if istat.statistic > 0 else [0]
            tvalues_neg += [istat.statistic] if istat.statistic < 0 else [0]
    print("Positive p-values: ", pvalues_pos)
    print("Negative p-values: ", pvalues_neg)
    return pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg


# Finding contiguous time cluster
def find_clusters(pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg):
    valid_window_pos = [i for i, v in enumerate(pvalues_pos) if v <= 0.05]
    valid_window_neg = [i for i, v in enumerate(pvalues_neg) if v <= 0.05]
    ## REMOVE FOR NULL FUNCTION
    print("Valid positive windows are: {0}\n".format(valid_window_pos))
    print("Valid negative windows are: {0}\n".format(valid_window_neg))

    # Obtain clusters (3 or more consecutive meaningful time)
    clusters_pos = [list(group) for group in mit.consecutive_groups(valid_window_pos)]
    clusters_pos = [group for group in clusters_pos if len(group) >= 2]

    clusters_neg = [list(group) for group in mit.consecutive_groups(valid_window_neg)]
    clusters_neg = [group for group in clusters_neg if len(group) >= 2]

    adj_clusters_pos = []
    for c in clusters_pos:
        new_list = [((x * 10) - 200) for x in c]
        adj_clusters_pos.append(new_list)

    adj_clusters_neg = []
    for c in clusters_neg:
        new_list = [((x * 10) - 200) for x in c]
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

    ## REMOVE FOR NULL FUNCTION
    print("Positive tmass values are: {0}\n".format(t_mass_pos))
    max_t_mass_pos = max(t_mass_pos)

    t_mass_neg = [0]
    for c in clusters_neg:
        t_scores_neg = 0
        for time in c:
            t_scores_neg += tvalues_neg[time]
        t_mass_neg += [t_scores_neg]

    ## REMOVE FOR NULL FUNCTION
    print("Negative tmass values are: {0}\n".format(t_mass_neg))
    max_t_mass_neg = min(t_mass_neg)

    max_abs_tmass = max(max_t_mass_pos, abs(max_t_mass_neg))

    ## REMOVE FOR NULL FUNCTION
    print("The max positive t mass is: {0}\n".format(max_t_mass_pos))
    print("The max negative t mass is: {0}\n".format(max_t_mass_neg))
    print("The max absolute t mass is: {0}\n".format(max_abs_tmass))

    return max_abs_tmass
