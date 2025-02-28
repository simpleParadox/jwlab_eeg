import pandas as pd
import numpy as np
import random
from scipy import stats
import more_itertools as mit
import time
import multiprocessing as mp
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import sys
import psutil



sys.path.insert(1, '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/classification/code')  ## For loading the following files.

from jwlab.ml_prep_perm import prep_ml, prep_matrices_avg, remove_samples
from matplotlib import pyplot as plt, cm
sys.path.insert(1, '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg')
from regression.functions import get_w2v_embeds_from_dict, two_vs_two, extended_2v2_phonemes, extended_2v2_perm, \
    get_phoneme_onehots, get_phoneme_classes, get_sim_agg_first_embeds, get_sim_agg_second_embeds, extended_2v2, \
    w2v_across_animacy_2v2, w2v_within_animacy_2v2, \
    ph_within_animacy_2v2, ph_across_animacy_2v2, get_audio_amplitude, get_stft_of_amp, \
    get_tuned_cbt_childes_w2v_embeds, \
    get_all_ph_concat_embeds, \
    get_glove_embeds, get_reduced_w2v_embeds, sep_by_prev_anim
from sklearn.linear_model import Ridge
from regression.rsa_helper import make_rdm, corr_between_rdms

labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}


minimal_mouth_labels_include = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                                6: 'duck', 10: 'cookie', 12: 'cup', 13: 'juice',
                                14: 'milk', 15: 'spoon'}

minimal_mouth_labels_exclude = {4: 'cat', 5: 'dog', 7: 'mom',
                                8: 'banana', 9: 'bottle', 11: 'cracker'}

first_sound_visible_on_face = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny', 7: 'mom', 8: 'banana', 9: 'bottle',
                               14: 'milk'}
not_fist_sound_visible_on_face = {4: ' cat', 5: 'dog', 6: 'duck', 10: 'cookie', 11: 'cracker', 12: 'cup', 13: 'juice',
                                  15: 'spoon'}


def reshape_to_60000(X, remove_avg=False):
    labels = X[0][0]['label']
    participants = X[0][0]['participant']
    # for i in range(len(X)):
    #
    temp = pd.concat(X[0], axis=1)
    temp = temp.drop(['label', 'participant'], axis=1)

    if remove_avg == True:
        avg = temp.iloc[:, :].mean(axis=0)
        temp1 = temp - avg

    t = pd.concat([temp1, labels, participants], axis=1)
    res = t.iloc[:, :-1].groupby(['label']).mean()

    return res

def plot_eeg_rsm(df):
    plt.clf()
    corr = df.T.corr()
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharey=True)
    # corr = mat
    mask = np.zeros_like(corr)
    labels = list(labels_mapping.values())
    mask[np.triu_indices_from(mask)] = True  # For printing only the lower triangle of the matrix.
    # sns.color_palette('coolwarm', as_cmap=True)
    sns.heatmap(corr, mask=mask, ax=ax[0],
                xticklabels=labels, yticklabels=labels, cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
                cbar_kws={'label': "Pearson Correlation"}, center=0, vmin=-0.64, vmax=0.57)
    # ax = sns.heatmap(corr, mask=mask,
    #                  xticklabels=labels, yticklabels=labels, cmap='coolwarm',
    #                  cbar_kws={'label': "Pearson Correlation"}, center=0, annot=True,
    #                  annot_kws={"fontsize":5})

    # ax2 = plt.twinx()
    sns.heatmap(corr, mask=mask, ax=ax[1],
                xticklabels=labels, yticklabels=labels, cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
                cbar_kws={'label': "Pearson Correlation"}, center=0, vmin=-0.64, vmax=0.57, annot=True,
                annot_kws={"fontsize": 7})

    fig.suptitle("9m 0-500 RSM avg_removed")
    plt.xlabel("Words")
    plt.ylabel("Words")
    plt.show()


def process_temp_results(data):
    # This function processes the temp_results and return the average for all the iterations of each window.
    temp_results_dict = [d[0] for d in data]
    d_list = []
    for i in range(len(temp_results_dict)):
        int_dict = {}
        for key, val in temp_results_dict[i].items():
            int_dict[key] = val[0]
        d_list.append(int_dict)
    res_df = pd.DataFrame(d_list)
    answer = dict(res_df.mean())
    answer_dict = {}
    for key, val in answer.items():
        answer_dict[key] = [val]
    return [answer_dict]


def cluster_analysis_procedure(age_group, useRandomizedLabel, averaging, sliding_window_config, cross_val_config,
                               type='simple'):
    print("Cluster analysis procedure")
    num_folds, cross_val_iterations, sampling_iterations = cross_val_config[0], cross_val_config[1], cross_val_config[2]

    results = {}
    animacy_results = {}
    tgm_results = []
    flag = 0
    preds_results = {}
    w2v_res_list = []
    cbt_res_list = []
    sampl_iter_word_pairs_2v2 = []
    for i in range(sampling_iterations):
        print("Sampling iteration: ", i)
        if averaging == "permutation":
            X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config,
                                                      downsample_num=1000)

            # X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel)  # This is a new statement.

            temp_results = cross_validaton(cross_val_iterations, num_win, num_folds, X, y)

            # temp_results = cross_validaton_averaging(X_train, X_test, y_train, y_test, useRandomizedLabel)

        elif averaging == "average_trials_and_participants":

            X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, "no_average_labels",
                                                      sliding_window_config, downsample_num=1000)

            # X = remove_samples(X)  # Use only for the 9 month olds.

            # X = minimal_mouth_X(X)

            # Done: Group 'X' across windows and then across stimuli for the same stimuli.
            # Function to encapsulate processing.
            # mat = reshape_to_60000(X, remove_avg=True)
            # # mat = eeg_group_by(X)
            # plot_eeg_rsm(mat)

            # monte_carlo_aniamcy_from_vectors()
            # break

            if type == 'permutation':
                # The split of the dataset into train and test set happens more than once here for each permuted label assignment.
                temp_results_list = []
                for i in range(50):
                    X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel,
                                                                         train_only=False, test_size=0.2)
                    temp_results, temp_animacy_results, temp_preds, temp_diag_tgm, word_pairs_2v2_sampl = cross_validaton_nested(X_train, y_train, X_test, y_test)
                    temp_results_list.append(temp_results)

                # Process temp_results_list to obtain a single "temp_results" list.
                temp_results = process_temp_results(temp_results_list)

            else:
                X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False,
                                                                     test_size=0.2)
                temp_results, temp_animacy_results, temp_preds, temp_diag_tgm, word_pairs_2v2_sampl = cross_validaton_nested(X_train, y_train, X_test, y_test)

            # temp_results, temp_diag_tgm = cv_residual_w2v_ph_eeg(X, age_group)

            # X = sep_by_prev_anim(X,y, current_type='animate', prev_type='animate')
            # X_train, X_test, y_train\
            #     , y_test = prep_matrices_avg(X, age_group, useRandomizedLabel)

            # temp_results, temp_diag_tgm = cv_residual_w2v_ph_eeg(X_train, X_test, y_train, y_test)
            # w2v_res, cbt_res = cv_all_ph_concat_padded_residual(X_train, X_test, y_train, y_test)
            #
            # w2v_res_list.append(w2v_res)
            # cbt_res_list.append(cbt_res)

            # For phonemes and w2v embeddings.
            # temp_results, temp_diag_tgm, word_pairs_2v2_sampl = cross_validaton_nested(X_train, y_train, X_test, y_test)
            #
            # For concatenation of w2v and phonemes, concats etc.
            # temp_results, temp_diag_tgm = cross_validaton_nested_concat(X_train, y_train, X_test, y_test)

            # sampl_iter_word_pairs_2v2.append(word_pairs_2v2_sampl)
            # Structure: [ {0:{0:[],1:[]}, 1:{0:[],1:[]}, ... }, {} ]     <- First dictionary is for the first sampling iteration and so on.
            # Done: Maybe draw a graph showing the word and change in accuracy over time.
            #
            #
            # _results.append(temp_diag_tgm)

            if sampling_iterations == 0:
                print("Warning: This does not do fold validation")
        elif averaging == "tgm":
            # Run this for within age group TGM.
            
            if type == 'permutation':
                # The split of the dataset into train and test set happens more than once here for each permuted label assignment.
                temp_results_list = []
                dataset_list = []
                # for i in range(5):
                #     X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel,
                #                                                          train_only=False, test_size=0.2)
                #     dataset_list.append([X_train, X_test, y_train, y_test])
                # dataset_list = tuple(dataset_list)
                # Implement multiprocessing here.
                cpus = 8
                pool = mp.Pool(cpus)#psutil.cpu_count(logical=False))
                p = [k for k in range(cpus)]
                res = pool.map(cross_validaton_tgm, tuple(p))

                # temp_results = cross_validaton_tgm(X_train, y_train, X_test, y_test)
                # tgm_results.append(temp_results)
                print(res)
                tgm_results.append(res.get())

            else:
                X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False, test_size=0.2)
                temp_results = cross_validaton_tgm(X_train, y_train, X_test, y_test)
                tgm_results.append(temp_results)
            # X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False,
            #                                                      test_size=0.2)
            # temp_results = cross_validaton_tgm(X_train, y_train, X_test, y_test)
            # tgm_results.append(temp_results)  # The temp_results is expected to be a square matrix.
        elif averaging == 'across':
            # Training on one group and testing on another group.
            age_group_1 = 9
            age_group_2 = 12
            X_1, y_1, good_trial_count_1, num_win_1 = prep_ml(age_group_1, useRandomizedLabel, "no_average_labels",
                                                              sliding_window_config, downsample_num=1000)

            X_2, y_2, good_trial_count_2, num_win_2 = prep_ml(age_group_2, useRandomizedLabel, "no_average_labels",
                                                              sliding_window_config, downsample_num=1000)

            X_train_1, X_test_1, y_train_1, y_test_1 = prep_matrices_avg(X_1, age_group_1, useRandomizedLabel,
                                                                         train_only=True, test_size=0)

            # Now the other group
            X_train_2, X_test_2, y_train_2, y_test_2 = prep_matrices_avg(X_2, age_group_2, useRandomizedLabel,
                                                                         train_only=False, test_size=1.0)
            temp_results = cross_validaton_tgm(X_train_1, y_train_1, X_test_2, y_test_2)
            tgm_results.append(temp_results)




        else:
            print("Warning: This will only use the requested averaging matrix to perform a cross val")
            X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config,
                                                      downsample_num=1000)

            temp_results = cross_validaton(cross_val_iterations, num_win, num_folds, X, y)

        # for i in range(len(temp_preds)):
        #     if i not in preds_results.keys():
        #         preds_results[i] = {}
        #     for j in range(len(temp_preds[i])):
        #         if j in preds_results[i].keys():
        #             preds_results[i][j] += temp_preds[i][j]
        #         else:
        #             preds_results[i][j] = temp_preds[i][j]
        # # For predicting animacy from predicting word embeddings.

        # for i in range(len(temp_animacy_results)):
        #     if i not in animacy_results.keys():
        #         animacy_results[i] = {}
        #     for j in range(len(temp_animacy_results[i])):
        #         if j in animacy_results[i].keys():
        #             animacy_results[i][j] += temp_animacy_results[i][j]
        #         else:
        #             animacy_results[i][j] = temp_animacy_results[i][j]

        # # if flag == 0:
        # for i in range(len(temp_results)):
        #     if i not in results.keys():
        #         results[i] = {}
        #     for j in range(len(temp_results[i])):
        #         if j in results[i].keys():
        #             results[i][j] += temp_results[i][j]
        #         else:
        #             results[i][j] = temp_results[i][j]
    # perm_results = np.array(results)
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # # filename = date + '.npz'
    # np.savez_compressed(f'G:\jw_lab\jwlab_eeg\\regression\permutation_test_results\\{timestr}.npz', perm_results)
    # preds_results = np.array(preds_results)
    # np.savez_compressed('' ,preds_results)
    # else:
    # Averaging was of type 'tgm'.
    # Calculate average of all the matrices.
    final_tgm = np.mean(tgm_results, axis=0)
    # Save the tgm in a csv file.
    step_size = sliding_window_config[3]
    ind = np.arange(-200, 1000, step_size).tolist()
    cols = np.arange(-200, 1000, step_size).tolist()
    df = pd.DataFrame(data=final_tgm, index=ind, columns=cols)
    df.to_csv(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/tgm_results/19-07-2021 {age_group}m tgm 100ms 10ms perm 50 iters.csv")

    # if flag == 0:

    # For predicting animacy from predictions.
    # pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg = t_test(animacy_results, num_win, num_folds)
    #
    # adj_clusters_pos, adj_clusters_neg, clusters_pos, clusters_neg = find_clusters(pvalues_pos, pvalues_neg,
    #                                                                                tvalues_pos, tvalues_neg)
    #
    # max_abs_tmass, t_mass_pos, t_mass_neg = get_max_t_mass(clusters_pos, clusters_neg, tvalues_pos, tvalues_neg)
    #
    #
    # # For predicting raw w2v embeddings from EEG.
    # pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg = t_test(results, num_win, num_folds)
    #
    # clusters_pos, clusters_neg = find_clusters(pvalues_pos, pvalues_neg, tvalues_pos, tvalues_neg)
    #
    # max_t_mass = get_max_t_mass(clusters_pos, clusters_neg, tvalues_pos, tvalues_neg)
    #
    # ## REMOVE FOR NULL FUNCTION
    # if len(sliding_window_config[2]) == 1:
    #     # createGraph(results, tvalues_pos, clusters_pos)
    #     # print(results)
    #     createGraph(animacy_results, tvalues_pos, clusters_pos)
    #     print(animacy_results)
    #     # word_pair_graph(sampl_iter_word_pairs_2v2)
    # else:
    #     print("Graph function is not supported for multiple window sizes")

    # rsa(np.mean(w2v_res_list, axis=0), np.mean(cbt_res_list, axis=0))
    return results


def process_pair_graph(data_dict):
    all_stim_bin_dict = {}
    window_len = len(data_dict[0])

    for wind in range(window_len):
        window_dict = {}
        all_stim_wind_avg = []
        for iter_dict in data_dict:  # 'iter_dict' is for each sampling iteration.
            window_word_dict = iter_dict[wind]  # {0: [], 1: [], ...}
            for key, value in window_word_dict.items():
                if key not in window_dict.keys():
                    window_dict[key] = value
                else:
                    window_dict[key].extend(value)

        # Now calculate the average.
        for k in range(0, 16):
            all_stim_wind_avg.append(np.mean(window_dict[k]))

        all_stim_bin_dict[wind] = all_stim_wind_avg
        # Now for each window we have a 2d array of size (should be) 16x1. The 16x1 array is averaged across 50 iterations for each stimuli.

    # Now process some more to get them lined up in one list. So finally we will have a 2d matrix where each row will be for one word.
    avg_pair_acc_dict = {}
    for stim in range(0, 16):
        avg_pair_acc_dict[stim] = []
        for window, acc in all_stim_bin_dict.items():
            avg_pair_acc_dict[stim].append(acc[stim])

    # Now we should have a 2d array where for each stimuli there will be a list containing the accuracies for each window.

    return avg_pair_acc_dict


def word_pair_graph(sampl_iter_word_pairs_2v2):
    # Structure -> [{0: {0: [], 1: [], ...}, 1: {0: [], 1: []}, ...}, {}]
    # First process the dictionary.
    avg_pair_acc_dict = process_pair_graph(sampl_iter_word_pairs_2v2)

    plt.clf()
    plt.figure()
    length_per_window_plt = 1200 / len(avg_pair_acc_dict[0])
    x_graph = np.arange(-200, 1000, length_per_window_plt)
    for stim, accs in avg_pair_acc_dict.items():
        if stim < 8:
            plt.plot(x_graph, accs, label=str(stim))
        else:
            break
    plt.legend(loc=1, bbox_to_anchor=(1.20, 1.0), title="12m Animate")
    plt.xlabel("Time (ms)")
    plt.ylabel("2v2 Accuracy")
    plt.title("Accuracy over time given word stimuli 12m (animate)")
    plt.savefig("12m pre-w2v animate stim comparison")

    plt.clf()
    plt.figure()
    for stim, accs in avg_pair_acc_dict.items():
        if stim >= 8:
            plt.plot(x_graph, accs, label=str(stim))
    plt.legend(loc=1, bbox_to_anchor=(1.20, 1.0), title="12m Inanimate")
    plt.xlabel("Time (ms)")
    plt.ylabel("2v2 Accuracy")
    plt.title("Accuracy over time given word stimuli 12m (inanimate)")
    plt.savefig("12m pre-w2v inanimate stim comparison")


def plt_rdm(mat, title, f_name):
    plt.clf()
    fig = plt.figure()
    # corr = mat
    mask = np.zeros_like(corr)
    labels = list(labels_mapping.values())
    mask[np.triu_indices_from(mask)] = True  # For printing only the lower triangle of the matrix.
    ax = sns.heatmap(corr, mask=mask,
                     xticklabels=labels, yticklabels=labels,
                     cmap="YlGnBu", cbar_kws={'label': "Dissimilarity \n Cosine Distance"})
    # im = ax.matshow(mat, cmap=mpl.cm.jet)
    # fig.colorbar(im)
    # ax.set(xticks=range(n), xticklabels=labels)
    # # ax.set_xticks(np.arange(n))
    # # ax.set_xticklabels(labels)
    # ax.set_yticks(np.arange(n))
    # ax.set_yticklabels(labels)

    # ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    # Rotate and align top ticklabels
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
    #          ha="left", va="center", rotation_mode="anchor")

    ax.set_title(title, pad=20)
    fig.tight_layout()
    # plt.show()

    fig.savefig(f_name)


def rsa(embeds_w2v, embeds_cbt):
    # Default metric is 'cosine'.
    rdm_w2v = make_rdm(embeds_w2v)
    rdm_cbt = make_rdm(embeds_cbt)

    plt_rdm(rdm_w2v, "W2V res all_ph rdm", "W2V res RDM from all_ph_concat")
    plt_rdm(rdm_cbt, "CBT_cdes res all_ph rdm", "CBT_chdes res RDM from all_ph_concat")

    # 'spearman' is default correlation metric.
    rdm_corr, p_val = corr_between_rdms(rdm_w2v, rdm_cbt)
    print(f"Correlation between w2v_embeds and cbt_embeds is {rdm_corr} with p-value: {p_val}")


def createGraph(results, t_mass_pos, adj_clusters_pos):
    scoreMean = []
    stdev = []
    for i in range(len(results)):
        for j in range(len(results[i])):
            scoreMean.append(round(np.mean(results[i][j]), 4))
            stdev.append(round(stats.sem(results[i][j]), 4))

    length_per_window_plt = 10  # 1200 / len(scoreMean)
    x_graph = np.arange(-200, 910, length_per_window_plt)
    x_graph += 50  # NOTE: Change this based on the window size.
    y_graph = scoreMean

    stdevplt = np.array(stdev)
    error = stdevplt

    # Done: Find the above chance accuracy x1 and x2 values for shading. Do you really need the tvalues_pos?
    # You need tvalues_pos if you are shading the region with max t-value. This might be a good idea.
    # It also might be a good idea to save the results somewhere to avoid rerunning the experiments everytime.
    # max_pos_tvalue_idx = t_mass_pos.index(max(t_mass_pos)) - 1
    # # # print("max_pos_tvalue_idx: ", max_pos_tvalue_idx)
    # # # print("clusters_pos: ", clusters_pos)
    # # print("adj_clusters_pos: ", adj_clusters_pos)
    x1 = 240 + 50  # adj_clusters_pos[max_pos_tvalue_idx][0] + 50
    x2 = 840 + 50  # adj_clusters_pos[max_pos_tvalue_idx][-1] + 50

    # Running the following line will fail if h1 and h0 are not initialized.
    # h0 needs to be initialized for many permutation iterations.
    reject, pvals_corrected, p_vals_idx_sort = significance(h1, h0)
    dot_idxs = p_vals_idx_sort[reject]
    x_dots = x_graph[dot_idxs]

    # Make sure to change this index to which you wanna retrieve.
    # pink_clusters = [list(i) for i in mit.consecutive_groups(sorted(x_graph[p_vals_idx_sort[:27]] / 10))]

    y_dots = [0.35] * len(x_dots)

    plt.clf()
    # plt.rcParams["figure.figsize"] = (20, 15)

    plt.plot(x_graph, y_graph, 'k-', label='9m')

    # , markersize=10)
    ylim = [0.3, 0.8]
    plt.ylim(ylim[0], ylim[1])
    # for cluster in clusters[1:]:
    #     x1 = cluster[0] * 10
    #     x2 = cluster[-1] * 10
    #     plt.fill_betweenx(y=ylim, x1=x1, x2=x2, color="#f5e6fc")
    plt.fill_betweenx(y=ylim, x1=x1, x2=x2, color="#f5e6fc")
    plt.axhline(0.5, linestyle='--', color='#696969')
    plt.axvline(0, linestyle='--', color='#696969')
    plt.fill_between(x_graph, y_graph - error, y_graph + error, color='#58c0fc')
    plt.scatter(x_dots, y_dots, marker='.')
    plt.title("9m 100-10 w2v_res_from_eeg perm 50iter shift-r 50ms")
    plt.xlabel("Time (ms)")
    plt.ylabel("2v2 Accuracy")
    plt.xticks(np.arange(-200, 1001, 200), ['-200', '0', '200', '400', '600', '800', '1000'])
    plt.legend(loc=1)
    acc_at_zero = y_graph[np.where(x_graph == 0)[0][0]]
    plt.text(700, 0.35, str(f"Acc At 0ms: {acc_at_zero}"))
    plt.show()

    plt.savefig("13-06-2021 avg_trials_and_ps anim_from_pred w2v only 9m 100ms 10ms 50iters non-perm")



def calculate_residual(true_vecs, pred_vecs):
    # Note: The arugments contain many arrays.
    return true_vecs - pred_vecs


def cv_all_ph_concat_padded_residual(X_train, X_test, y_train, y_test):
    # First calculate the residuals. Train on ph, test on w2v, then get w2v residuals.
    # scoring = 'accuracy'
    scoring = 'neg_mean_squared_error'

    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    #
    i = 0
    j = 0

    y_train_w2v = get_w2v_embeds_from_dict(y_train[i][j])
    y_test_w2v = get_w2v_embeds_from_dict(y_test[i][j])

    y_train_cbt = get_cbt_childes_w2v_embeds(y_train[i][j])
    y_test_cbt = get_cbt_childes_w2v_embeds(y_test[i][j])

    x_train_ph = get_all_ph_concat_embeds(y_train[i][j])
    x_test_ph = get_all_ph_concat_embeds(y_test[i][j])
    model = Ridge()

    clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=12, cv=5)
    clf.fit(x_train_ph, y_train_w2v)
    y_pred_w2v_test = clf.predict(x_test_ph)  # Get the prediction w2v embeddings from the phonemes.
    # y_pred_w2v_train = clf.predict(x_train_ph)

    clf_cbt = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=-1, cv=5)
    clf_cbt.fit(x_train_ph, y_train_cbt)
    y_pred_cbt_test = clf.predict(x_test_ph)
    # y_pred_cbt_train = clf.predict(x_train_ph)

    # Now we calculate residual for training and test data.
    # w2v_train_res = calculate_residual(y_train_w2v, y_pred_w2v_train)
    w2v_test_res = calculate_residual(y_test_w2v, y_pred_w2v_test)

    cbt_test_res = calculate_residual(y_test_cbt, y_pred_cbt_test)

    return w2v_test_res, cbt_test_res


def cv_residual_w2v_ph_eeg(X_train, X_test, y_train, y_test):
    # First calculate the residuals. Train on ph, test on w2v, then get w2v residuals.
    results = []
    tgm_matrix_temp = np.zeros((120, 120))
    # scoring = 'accuracy'
    scoring = 'neg_mean_squared_error'

    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    #
    i = 0
    j = 0

    y_train_w2v = get_w2v_embeds_from_dict(y_train[i][j])
    y_test_w2v = get_w2v_embeds_from_dict(y_test[i][j])

    x_train_ph = get_all_ph_concat_embeds(y_train[i][j])
    x_test_ph = get_all_ph_concat_embeds(y_test[i][j])
    model = Ridge()

    clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=12, cv=5)

    clf.fit(x_train_ph, y_train_w2v)

    y_pred_w2v_test = clf.predict(x_test_ph)  # Get the prediction w2v embeddings.

    y_pred_w2v_train = clf.predict(x_train_ph)

    # Now we calculate residual for training and test data.
    w2v_train_res = calculate_residual(y_train_w2v, y_pred_w2v_train)
    w2v_test_res = calculate_residual(y_test_w2v, y_pred_w2v_test)

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

            # y_train_w2v = get_w2v_embeds_from_dict(y_train[i][j])
            # y_test_w2v = get_w2v_embeds_from_dict(y_test[i][j])

            # One-hot vectors here.
            # y_train_labels = get_phoneme_classes(y_train[i][j])
            # y_test_labels = get_phoneme_classes(y_test[i][j])

            # Get first sim_agg embeddings here.
            # x_train_ph = get_sim_agg_first_embeds(y_train[i][j])
            # x_test_ph = get_sim_agg_first_embeds(y_test[i][j])
            # which_phoneme = 1

            # Get second sim_agg embeddings here
            # y_train_labels = get_sim_agg_second_embeds(y_train[i][j])
            # y_test_labels = get_sim_agg_second_embeds(y_test[i][j])
            # which_phoneme = 2

            # model = Ridge()

            clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=12, cv=5)

            # clf.fit(x_train_ph, y_train_w2v)

            # y_pred_w2v_test = clf.predict(x_test_ph)  # Get the prediction w2v embeddings.
            #
            # y_pred_w2v_train = clf.predict(x_train_ph)

            # Now we calculate residual for training and test data.
            # w2v_train_res = calculate_residual(y_train_w2v, y_pred_w2v_train)
            # w2v_test_res = calculate_residual(y_test_w2v, y_pred_w2v_test)

            # Now we train on EEG to predict the residuals from Word2Vec embeddings which were predicted from the phoneme embeddings.
            model_res = Ridge()

            clf_res = GridSearchCV(model_res, ridge_params, scoring=scoring, n_jobs=12, cv=5)

            clf_res.fit(X_train[i][j], w2v_train_res)

            y_pred_w2v_res = clf_res.predict(X_test[i][j])

            points, total_points, testScore, gcf, grid = extended_2v2(w2v_test_res, y_pred_w2v_res)
            # points, total_points, testScore, gcf, grid = w2v_across_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid= w2v_within_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid = extended_2v2_phonemes(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)

            # testScore = accuracy_score(y_test_labels, y_pred)

            tgm_matrix_temp[j, j] = testScore

            if j in temp_results.keys():
                temp_results[j] += [testScore]
            else:
                temp_results[j] = [testScore]

        results.append(temp_results)

    return results, tgm_matrix_temp


def cross_validaton_nested(X_train, y_train, X_test, y_test):
    results = []
    preds = []
    animacy_results = []
    tgm_matrix_temp = np.zeros((120, 120))
    # scoring = 'accuracy'
    scoring = 'neg_mean_squared_error'

    ## Define the hyperparameters.
    # ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    ridge_params = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}
    #
    all_word_pairs_2v2 = {}
    for i in range(len(X_train)):
        temp_results = {}
        temp_preds = {}
        temp_animacy_results = {}
        for j in range(len(X_train[i])):

            # this is for predicting the second phoneme only (sim_agg.csv).
            # First remove the data for which the second phoneme is not present.
            # NOTE: The remove data function is not being used because phoneme alternatives are now being used.
            # X_train[i][j], y_train[i][j] = remove_data(X_train[i][j], y_train[i][j])
            # X_test[i][j], y_test[i][j] = remove_data(X_test[i][j], y_test[i][j])

            # model = SVC(kernel = 'rbf', C=1e-9, gamma = .0001)
            # model = LinearSVC(C=1e-9, max_iter=1000)

            # Get tuned w2v vectors on CBT and Childes
            y_train_labels_w2v = get_tuned_cbt_childes_w2v_embeds(y_train[i][j])
            y_test_labels_w2v = get_tuned_cbt_childes_w2v_embeds(y_test[i][j])

            # Get google news pretrained word vectors.
            y_train_labels_w2v = get_w2v_embeds_from_dict(y_train[i][j])
            y_test_labels_w2v = get_w2v_embeds_from_dict(y_test[i][j])

            # The following functions retrieve the embeddings of size 16x16 (SVD/PCA).
            # y_train_labels_w2v = get_reduced_w2v_embeds(y_train[i][j], type='svd')
            # y_test_labels_w2v = get_reduced_w2v_embeds(y_test[i][j], type='svd')

            # Get glove embeddings here.
            # y_train_labels_w2v = get_glove_embeds(y_train[i][j])
            # y_test_labels_w2v = get_glove_embeds(y_test[i][j])

            # One-hot vectors here.
            # y_train_labels = get_phoneme_classes(y_train[i][j])
            # y_test_labels = get_phoneme_classes(y_test[i][j])

            # Get first sim_agg embeddings here.
            # y_train_labels_ph = get_sim_agg_first_embeds(y_train[i][j])
            # y_test_labels_ph = get_sim_agg_first_embeds(y_test[i][j])
            # which_phoneme = 1

            # Get second sim_agg embeddings here
            # y_train_labels_ph = get_sim_agg_second_embeds(y_train[i][j])
            # y_test_labels_ph = get_sim_agg_second_embeds(y_test[i][j])
            # which_phoneme = 2

            # Get fourier transform of audio amplitudes here.
            # y_train_labels_audio = get_audio_amplitude(y_train[i][j])
            # y_test_labels_audio = get_audio_amplitude(y_test[i][j])

            # get stft of audio applitudes here.
            # y_train_labels_audio_stft = get_stft_of_amp(y_train[i][j])
            # y_test_labels_audio_stft = get_stft_of_amp(y_test[i][j])

            # model = LogisticRegression(multi_class='multinomial')

            model = Ridge()

            clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=-1, cv=5)

            # clf.fit(X_train[i][j], y_train_labels_w2v)
            clf.fit(X_train[i][j], y_train_labels_w2v)
            y_pred = clf.predict(X_test[i][j])

            # Many scoring functions.

            # points, total_points, testScore, gcf, grid, word_pairs = extended_2v2(y_test_labels_w2v, y_pred)
            # points, total_points, testScore, gcf, grid = w2v_across_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid= w2v_within_animacy_2v2(y_test_labels, y_pred)
            # points, total_points, testScore, gcf, grid = extended_2v2_phonemes(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)

            # Across and within for phonemes
            # points, total_points, testScore, gcf, grid = ph_across_animacy_2v2(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)
            # points, total_points, testScore, gcf, grid = ph_within_animacy_2v2(y_test_labels, y_pred, y_test[i][j], first_or_second=which_phoneme)

            # Using word embeddings to predict animacy.
            # animacy_score = monte_carlo_animacy_from_vectors(y_pred)

            # all_word_pairs_2v2[j] = word_pairs  # Here 'j' is each window across the whole timeline.

            # testScore = accuracy_score(y_test_labels, y_pred)

            # tgm_matrix_temp[j, j] = testScore

            if j in temp_preds.keys():
                temp_preds[j] += [y_pred]
            else:
                temp_preds[j] = [y_pred]

            # if j in temp_animacy_results.keys():
            #     temp_animacy_results[j] += [animacy_score]
            # else:
            #     temp_animacy_results[j] = [animacy_score]

            # if j in temp_results.keys():
            #     temp_results[j] += [testScore]
            # else:
            #     temp_results[j] = [testScore]

        # results.append(temp_results)
        preds.append(temp_preds)
        # animacy_results.append(temp_animacy_results)

    return results, animacy_results, preds, tgm_matrix_temp, all_word_pairs_2v2


# def cross_validaton_tgm(X_train, y_train, X_test, y_test):
def cross_validaton_tgm(p):
    # results = []
    age_group = 9
    useRandomizedLabel = True
    dataset_list = []
    sliding_window_config = [-200, 1000, [100], 10]
    X, y, good_trial_count, num_win = prep_ml(age_group, useRandomizedLabel, "no_average_labels",
                                                      sliding_window_config, downsample_num=1000)
    for i in range(5):
        X_train, X_test, y_train, y_test = prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False, test_size=0.2)
        dataset_list.append([X_train, y_test, X_test, y_test])

    tgm_matrix_temp = np.zeros((120, 120))
    print('Data loaded')

    ## Define the hyperparameters.
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

    xval_results = []
    for X_train, X_test, y_train, y_test in dataset_list:
        print('Process: ', p)
        for i in range(len(X_train)):
            temp_results = {}
            for j in range(len(X_train[i])):

                y_train_labels = get_w2v_embeds_from_dict(y_train[i][j])
                model = Ridge()
                clf = GridSearchCV(model, ridge_params, scoring='neg_mean_squared_error', n_jobs=2, cv=5)
                clf.fit(X_train[i][j], y_train_labels)

                for k in range(len(X_test[i])):
                    y_test_labels = get_w2v_embeds_from_dict(y_test[i][k])
                    y_pred = clf.predict(X_test[i][k])
                    points, total_points, testScore, gcf, grid, word_pairs = extended_2v2(y_test_labels, y_pred)

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
                # results.append(temp_results
        xval_results.append(tgm_matrix_temp)
    # return tgm_matrix_temp
    return xval_results


def average_fold_accs(data):
    result = {}
    for key, val in data.items():
        result[key] = np.mean(val)

    return result


def significance(h1, h0):
    """"
    This function calculates the p-value of the alternate hypothesis against the
    null distribution.
    """
    """h1 is the alternate hypothesis. This contains the observed values for
    each window which will then be averaged.
    h0 is the null hypothesis. h0 will have many values for one single window.
    The function performs the statistical test with the t-test with the bejamini hochberg correction.
    """
    # First we process 'h1' to average all the cross-validation folds accuracies for each window.
    null_h = h0
    alt_h = h1[0]

    alt_h_avg = average_fold_accs(alt_h)

    # Now we perform the significance testing between the 'h1_avg' and 'h0'.
    """
    The p-value is calculated by finding the number of times the permuted accuracies
    are above the observed value (true value).
    The process is done for each window.
    """

    denom = len(null_h[0])
    p_values_list = []  # Stores the window based p-values against the null distribution.
    for window in range(len(alt_h_avg)):
        obs_score = alt_h_avg[window]
        permute_scores = null_h[window]
        count = 0
        # Now count how many of the permute scores are >= obs_score.
        for j in range(len(permute_scores)):
            if permute_scores[j] > obs_score:
                count += 1

        p_value = count / denom
        p_values_list.append(p_value)

    # Implementing the Benjamini-Hochberg correction.
    # First have an index_array just in case.
    idxs = [i for i in range(len(alt_h_avg))]
    # Sort the p_values and idxs in ascending order.
    p_vals_list_asc, p_vals_idx_sort = (list(t) for t in zip(*sorted(zip(p_values_list, idxs))))
    p_vals_asc_rank = [i for i in range(len(alt_h_avg))]

    reject, pvals_corrected, alph_sidak, alph_bonf = multipletests(p_vals_list_asc, is_sorted=True, method='fdr_bh')

    p_vals_idx_sort = np.array(p_vals_idx_sort)

    return reject, pvals_corrected, p_vals_idx_sort









def get_permuted_val(results, ttest=False):
    # h0 = np.load('G:\jw_lab\jwlab_eeg\Results\9m_prew2v_from_egg_null_dist_100x50iters.npz', allow_pickle=True)
    # h0 = h0['arr_0']
    # h0 = h0.tolist()

    null_h = h0
    alt_h = results

    alt_h_avg = average_fold_accs(alt_h)

    if ttest == True:
        null_h_avg = average_fold_accs(null_h)

        return null_h_avg

    denom = len(null_h[0])
    p_values_list = []  # Stores the window based p-values against the null distribution.
    for window in range(len(alt_h_avg)):
        obs_score = alt_h_avg[window]
        permute_scores = null_h[window]
        count = 0
        # Now count how many of the permute scores are >= obs_score.
        for j in range(len(permute_scores)):
            if permute_scores[j] > obs_score:
                count += 1

        p_value = count / denom
        p_values_list.append(p_value)

    return p_values_list


def cluster_permutation_test(results):
    pvalues_pos = get_permuted_val(results[0], ttest=False)

    # Find clusters.
    valid_window_pos = [i for i, v in enumerate(pvalues_pos) if v < 0.05]

    clusters_pos = [list(group) for group in mit.consecutive_groups(valid_window_pos)]
    clusters_pos = [group for group in clusters_pos if len(group) >= 3]

    adj_clusters_pos = []
    for c in clusters_pos:
        new_list = [((x * 10) - 200) for x in c]
        adj_clusters_pos.append(new_list)

    return adj_clusters_pos


def t_test(results, num_win, num_folds):
    pvalues_pos = []
    pvalues_neg = []
    tvalues_pos = []
    tvalues_neg = []
    null_distribution_vals = get_permuted_val(results[0], ttest=True)
    for i in range(len(results)):
        for j in range(len(results[i])):
            # change the second argument below for comparison
            # Retrieve the permutation test values here.
            istat = stats.ttest_1samp(results[i][j], null_distribution_vals[j])
            istat = stats.ttest_1samp(results[i][j], null_distribution_vals[j])
            # istat = stats.ttest_1samp(results[i][j], 0.5)
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
    valid_window_pos = [i for i, v in enumerate(pvalues_pos) if v < 0.05]
    valid_window_neg = [i for i, v in enumerate(pvalues_neg) if v < 0.05]
    ## REMOVE FOR NULL FUNCTION
    print("Valid positive windows are: {0}\n".format(valid_window_pos))
    print("Valid negative windows are: {0}\n".format(valid_window_neg))

    # Obtain clusters (2 or more consecutive meaningful time) -> Initally it was 3 clusters. I changed it to 2 and now it's 3 again as of 09-04-2021.
    clusters_pos = [list(group) for group in mit.consecutive_groups(valid_window_pos)]
    clusters_pos = [group for group in clusters_pos if len(group) >= 3]

    clusters_neg = [list(group) for group in mit.consecutive_groups(valid_window_neg)]
    clusters_neg = [group for group in clusters_neg if len(group) >= 3]

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

    return adj_clusters_pos, adj_clusters_neg, clusters_pos, clusters_neg


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

    return max_abs_tmass, t_mass_pos, t_mass_neg
