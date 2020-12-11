import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine
from copy import deepcopy
import pandas as pd
import pickle
import random
import gensim
from numpy import load, savez_compressed
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import platform

os_name = platform.system()

w2v_path = None
avg_w2v_path = None
if os_name == 'Windows':
    w2v_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\all_w2v_embeds.npz"
    avg_w2v_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\all_w2v_embeds_avg_trial.npz"
    gen_w2v_all_ps_avg_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\gen_w2v_embeds_avg_trial_and_ps.npz"
    embeds_with_label_path = 'G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\embeds_with_label_dict.npz'
    bof_embeds_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\bof_w2v_embeds.npz"
    ph_embeds_path = "G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\phoneme_embeds.npz"
    ph_classes_path = "G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\phoneme_classes.npz"
    ph_first_one_hots_path = "G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\first_one_hots.npz"
    ph_second_one_hots_path = "G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\second_one_hots.npz"
    ph_second_classes_path = "G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\second_phoneme_classes.npz"
    ph_similarity_agg_path = "G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_data\\similarity_aggregated.csv"
    sim_agg_first_embeds_path = "G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\first_sim_agg_embeddings.npz"
    sim_agg_second_embeds_path = "G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\second_sim_agg_embeddings.npz"
elif os_name == 'Linux':
    w2v_path = os.getcwd() + "/regression/w2v_embeds/all_w2v_embeds.npz"
    avg_w2v_path = os.getcwd() + "/regression/w2v_embeds/all_w2v_embeds_avg_trial.npz"
    gen_w2v_all_ps_avg_path = os.getcwd() + "/regression/w2v_embeds/gen_w2v_embeds_avg_trial_and_ps.npz"
    embeds_with_label_path = os.getcwd() + "/regression/w2v_embeds/embeds_with_label_dict.npz"
    bof_embeds_path = os.getcwd() + "/regression/w2v_embeds/bof_w2v_embeds.npz"
    ph_embeds_path = os.getcwd() + "/regression/phoneme_embeddings/phoneme_embeds.npz"
    ph_classes_path = os.getcwd() + "/regression/phoneme_embeddings/phoneme_classes.npz"
    ph_first_one_hots_path = os.getcwd() + "/regression/phoneme_embeddings/first_one_hots.npz"
    ph_second_one_hots_path = os.getcwd() + "/regression/phoneme_embeddings/second_one_hots.npz"
    ph_second_classes_path = os.getcwd() + "/regression/phoneme_embeddings/second_phoneme_classes.npz"
    ph_similarity_agg_path = os.getcwd() + "/regression/phoneme_data/similarity_aggregated.csv"
    sim_agg_first_embeds_path = os.getcwd() + "/regression/phoneme_embeddings/first_sim_agg_embeddings.npz"
    sim_agg_second_embeds_path = os.getcwd() + "/regression/phoneme_embeddings/second_sim_agg_embeddings.npz"

word_list = ["baby", "BAD_STRING", "bird", "BAD_STRING", "cat", "dog", "duck", "mommy",
             "banana", "bottle", "cookie", "cracker", "BAD_STRING", "juice", "milk", "BAD_STRING"]

labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}

first_phonemes_mapping = {'baby':17, 'bear':17, 'bird':17, 'bunny':17,
                                 'cat':30, 'dog':19, 'duck':19, 'mom': 33,
                                 'banana': 17, 'bottle': 17, 'cookie': 30, 'cracker': 30,
                                 'cup': 30, 'juice': 22, 'milk': 33, 'spoon': 40}

stimuli_to_second_ipa_mapping = {'baby': 23, 'bear': 25, 'bird': 24, 'bunny': 15,
                                 'cat': 16, 'dog': 15, 'duck': 15, 'mom': 24,
                                 'banana': 24, 'bottle': 15, 'cookie': 37, 'cracker': 39,
                                 'cup': 15, 'juice': 44, 'milk': 25, 'spoon': 38}


def get_embeds_list():
    w2v_array = []
    file = np.load(embeds_with_label_path, allow_pickle=True)
    data = file['arr_0'][0]
    for i in range(16):
        w2v_array.append(data[i].tolist())

    return w2v_array


w2v_array = get_embeds_list()


def test_model_permute(X, y):
    # print("Test model permute")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90)
    model = Ridge()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # print(model.score(X_test, y_test))
    a, b, c = two_vs_two(y_test, preds)
    # print(c)
    return c


def test_model(X, y, labels_dict=None):
    # print("New test?")
    # print("Test model")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90)
    # print(X_train.shape)
    # print(y_train.shape)
    model = Ridge(tol=1e-4)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # print(model.score(X_test, y_test))
    # check_and_assign_labels(y_test, preds, labels_dict)
    a, b, c = two_vs_two(y_test, preds)

    # print(a)
    # print(b)
    # print(c)
    return c


def check_and_assign_labels(ytest, preds, labels_dict):
    """
    For y_test assign the correct keys as numbers from the labels_dict.
    For preds, do the same.
    """
    ytest_numbers = []
    preds_numbers = []
    for i in range(len(ytest)):
        for key, value in labels_dict.items():
            print("type(ytest[i]): ", type(ytest[i]))
            print("type(value): ", type(value))
            if np.array_equal(ytest[i], value):
                ytest_numbers.append(key)
            if np.array_equal(preds[i], value):
                preds_numbers.append(key)


def average_trials(df):
    num_participants = int(df.participant.max()) + 1
    print(num_participants)
    num_words = len(word_list)

    new_data = np.zeros((num_participants * num_words, len(df.columns) - 2))
    df_data = df.drop(columns=['label', 'participant'], axis=1)
    new_y = np.zeros(num_participants * num_words)
    participants = np.zeros(num_participants * num_words)

    for p in range(num_participants):
        for w in range(num_words):
            means = df_data[np.logical_and(df.participant == p, df.label == w)].values.mean(axis=0) if df_data[
                                                                                                           np.logical_and(
                                                                                                               df.participant == p,
                                                                                                               df.label == w)].size != 0 else 0
            print("Means: ", means)
            new_data[p * num_words + w, :] = means
            new_y[p * num_words + w] = -1 if np.isnan(means).any() else w
            participants[p * num_words + w] = p

    return new_data, new_y, participants, np.copy(new_y)


def average_trials_mod(df):
    all_avg_df = pickle.load(open('G:\\jw_lab\\jwlab_eeg\\regression\\data\\avg_trials_with_lab_and_ps.pkl', 'rb'))
    # Do some preprocessing here.
    t_df = all_avg_df[:208].copy()
    n_df = all_avg_df[208:].copy()
    # Now change labels for n_df.
    n_ps_len = len(n_df['participant'])
    new_ps_list = []
    for _ in range(21):
        for new_ps in range(16):
            new_ps_list.append(int(_))
    n_df['participant'] = new_ps_list
    data_up, y, participants_rt, w = average_trials(n_df)

    num_participants = n_df.participant.max() + 1
    num_words = len(word_list)

    new_data = np.zeros((num_participants * num_words, len(n_df.columns) - 2))
    df_data = n_df.drop(columns=['label', 'participant'], axis=1)
    new_y = np.zeros(num_participants * num_words)
    participants = np.zeros(num_participants * num_words)

    for p in range(num_participants):
        for w in range(num_words):
            means = df_data[np.logical_and(n_df.participant == p, n_df.label == w)].values.mean(axis=0) if df_data[
                                                                                                               np.logical_and(
                                                                                                                   n_df.participant == p,
                                                                                                                   n_df.label == w)].size != 0 else 0
            # print("Means: ", means)
            new_data[p * num_words + w, :] = means
            new_y[p * num_words + w] = -1 if np.isnan(means).any() else w
            participants[p * num_words + w] = p

    return new_data, new_y, participants, np.copy(new_y)


def average_trials_and_participants(df, participants):
    num_words = len(word_list)
    data, y, participants_rt, w = average_trials(df)

    new_data = np.zeros((num_words, len(df.columns) - 2))
    new_y = np.zeros(num_words)
    for w in range(num_words):
        count = 0
        for p in range(13):
            count += data[p * num_words + w]
        mean = count / len(participants)
        new_data[w, :] = mean
        new_y[w] = -1 if np.isnan(mean).any() else w
    new_data = new_data[new_y != -1, :]
    new_y = new_y[new_y != -1]
    new_data_df = pd.DataFrame(new_data)
    new_data_df.to_pickle('G:\\jw_lab\\jwlab_eeg\\regression\\data\\avg_trials_and_ps_13m.pkl')
    return new_data, new_y, np.ones(new_y.shape[0]) * -1, np.copy(new_y)


def two_vs_two_test(y_test, preds):
    # print("Ytest", y_test)
    # print("Preds",preds)
    points = 0
    total_points = 0
    for i in range(preds.shape[0] - 1):
        s_i = y_test[i]
        s_j = y_test[i + 1]
        s_i_pred = preds[i]
        s_j_pred = preds[i + 1]
        # print("S_i", s_i)
        # print('s_i_pred', s_i_pred)
        dsii = cosine_similarity([[s_i]], [[s_i_pred]])
        dsjj = cosine_similarity([[s_j]], [[s_j_pred]])
        dsij = cosine_similarity([[s_i]], [[s_j_pred]])
        dsji = cosine_similarity([[s_j]], [[s_i_pred]])
        # print("dsii: ", dsii)
        # print("dsjj: ", dsjj)
        # print("dsij: ", dsij)
        # print("dsji: ", dsji)
        if (dsii + dsjj) >= (dsij + dsji):
            points += 1
        total_points += 1
    return points, total_points, points / total_points


def get_idx_in_list(elem):
    return w2v_array.index(elem)


def get_word_pairs_by_key_pair(key_pairs):
    keys = key_pairs.split('_')
    word1 = labels_mapping[int(keys[0])]
    word2 = labels_mapping[int(keys[1])]
    return word1, word2, int(keys[0]), int(keys[1])


def plot_grid(grid):
    plt.matshow(grid)
    plt.colorbar()
    return plt.gcf()


def two_vs_two(y_test, preds):
    # print("Ytest", y_test[0])
    # print("Preds",preds[0])
    points = 0
    total_points = 0
    diff = []
    sum_ii_jj = []
    sum_ij_ji = []
    x_length = [_ for _ in range(preds.shape[0] - 1)]
    word_pairs = dict()
    index_pairs = []
    for i in range(preds.shape[0] - 1):
        s_i = y_test[i]
        s_j = y_test[i + 1]
        s_i_pred = preds[i]
        s_j_pred = preds[i + 1]
        # print("S_i", s_i)
        # print('s_i_pred', s_i_pred)
        dsii = cosine(s_i, s_i_pred)
        dsjj = cosine(s_j, s_j_pred)

        dsij = cosine(s_i, s_j_pred)
        dsji = cosine(s_j, s_i_pred)
        # print("dsii: ", dsii)
        # print("dsii abs: ", np.abs(dsii[0][0]))
        # print("dsij: ", dsij)
        # print("dsji: ", dsji)
        # # print("Addition", dsii+dsjj)
        # sum_ii_jj.append((dsii + dsjj))
        # sum_ij_ji.append((dsij + dsji))
        # diff.append((dsii + dsjj) - (dsij + dsji))
        if dsii + dsjj <= dsij + dsji:
            points += 1
            # si_idx = get_idx_in_list(s_i.tolist())
            # sj_idx = get_idx_in_list(s_j.tolist())
            # index_pairs.append([si_idx, sj_idx])
            # if f"{si_idx}_{sj_idx}" in word_pairs:
            #     word_pairs[f'{si_idx}_{sj_idx}'] += 1
            # else:
            #     word_pairs[f'{si_idx}_{sj_idx}'] = 1

        total_points += 1

    # si_idx = get_idx_in_list(s_i.tolist())
    # sj_idx = get_idx_in_list(s_j.tolist())
    # if f"{si_idx}_{sj_idx}" in word_pairs:
    #     word_pairs[f'{si_idx}_{sj_idx}'] += 1
    # else:
    #     word_pairs[f'{si_idx}_{sj_idx}'] = 1

    # max_value = max(word_pairs.values())
    # for key, val in word_pairs.items():
    #     if val == max_value:
    #         print(f'{val} {key} {get_word_pairs_by_key_pair(key)}')

    # diff_mean = np.mean(diff)
    # diff = np.array(diff)
    # below_zero_diff = diff[diff < 0]
    # above_zero_diff = diff[diff > 0]
    # above_zero_diff_mean = np.mean(above_zero_diff)
    # below_zero_diff_mean = np.mean(below_zero_diff)
    # sum_ii_jj_mean = np.mean(sum_ii_jj)
    # sum_ij_ji_mean = np.mean(sum_ij_ji)
    #
    # print("Diff mean ", diff_mean)
    # print("Below zero mean ", below_zero_diff_mean)
    # print("Above zero mean ", above_zero_diff_mean)
    # print("Sum ii jj ", sum_ii_jj_mean)
    # print("Sum ij ji ", sum_ij_ji_mean)
    #
    #
    # ## The following piece of code plots graphs for the difference between the sum of the cosine distances.
    # print("Points -> ", points)
    # print("Total points -> ", total_points)
    # plt.rcParams["figure.figsize"] = (20, 10)
    # plt.scatter(x_length, sum_ii_jj, color='b', label='correct', marker='o')
    # plt.scatter(x_length, sum_ij_ji, color='r', label='alternate', marker='v')
    # plt.plot(x_length, diff, color='g', label='Difference')
    # plt.text(max(x_length),-1.1, "Points below 0 line: "+str(points) + ",\nPoints above 0 line " + str(total_points-points))
    # plt.xlabel("Test sample")
    # plt.ylabel("Difference (correct - alternate)")
    # plt.axhline(y=0)
    # plt.legend(loc='upper left')
    # plt.title("Permuted labels")
    # plt.show()
    # plt.savefig('G:\jw_lab\jwlab_eeg\\regression\\test.png')
    ######
    ## Create the 16x16 graph for the different time windows. For each time window, you will have the y_test and preds.
    # First create a matrix of size 16 x 16.
    grid = np.zeros((16, 16))
    # for pair in index_pairs:
    #     row, col = pair
    #     # print(pair)
    #     grid[row, col] += 1

    # Next, for each word pair in the 2v2 test, increament that cell by 1.
    # But first, you need to find out the word pair. One way is to store the word
    # pairs in an array; in other words, store the index pairs.
    gcf = None  # plot_grid(grid)
    return points, total_points, points / total_points, gcf, grid



def extended_2v2(y_test, preds):
    """
    There are two additions to this function over the previous two_vs_two test.
    1. The grid figures will be symmetric now.
    2. The 16 samples in the test set will now be extended to 16C2=120 samples - without repetition.
    """
    points = 0
    total_points = 0
    # diff = []
    # sum_ii_jj = []
    # sum_ij_ji = []
    # x_length = [_ for _ in range(preds.shape[0] - 1)]
    # word_pairs = dict()
    # index_pairs = []
    for i in range(preds.shape[0] - 1):
        s_i = y_test[i]
        s_i_pred = preds[i]
        for j in range(i+1, preds.shape[0]):
            s_j = y_test[j]
            s_j_pred = preds[j]

            dsii = cosine(s_i, s_i_pred)
            dsjj = cosine(s_j, s_j_pred)
            dsij = cosine(s_i, s_j_pred)
            dsji = cosine(s_j, s_i_pred)

            # sum_ii_jj.append((dsii + dsjj))
            # sum_ij_ji.append((dsij + dsji))
            # diff.append((dsii + dsjj) - (dsij + dsji))

            if dsii + dsjj <= dsij + dsji:
                points += 1
                # si_idx = get_idx_in_list(s_i.tolist())
                # sj_idx = get_idx_in_list(s_j.tolist())
                # index_pairs.append([si_idx, sj_idx])
                # if f"{si_idx}_{sj_idx}" in word_pairs:
                #     word_pairs[f'{si_idx}_{sj_idx}'] += 1
                # else:
                #     word_pairs[f'{si_idx}_{sj_idx}'] = 1

            total_points += 1

    grid = np.zeros((16, 16))
    # for pair in index_pairs:
        # row, col = pair
        # # print(pair)
        # grid[row, col] += 1
        # grid[col, row] += 1

    # Next, for each word pair in the 2v2 test, increament that cell by 1. Have to make sure that the matrices are symmetric.
    # But first, you need to find out the word pair. One way is to store the word
    # pairs in an array; in other words, store the index pairs.

    gcf = None  # plot_grid(grid)
    return points, total_points, points / total_points, gcf, grid


# Added 05-12-2020
def w2v_across_animacy_2v2(y_test, preds):
    """
    The function compares the first 8 words(animate) with the last 8 words(inanimate).
    Note: There are a total of 16 words in the test set. All the word pairs are used.
    """
    # Get mid point.
    mid_idx = len(y_test) // 2
    points = 0
    total_points = 0
    for i in range(mid_idx):
        s_i = y_test[i]
        s_i_pred = preds[i]
        for j in range(mid_idx, len(y_test)):
            s_j = y_test[j]
            s_j_pred = preds[j]

            dsii = cosine(s_i, s_i_pred)
            dsjj = cosine(s_j, s_j_pred)
            dsij = cosine(s_i, s_j_pred)
            dsji = cosine(s_j, s_i_pred)

            # sum_ii_jj.append((dsii + dsjj))
            # sum_ij_ji.append((dsij + dsji))
            # diff.append((dsii + dsjj) - (dsij + dsji))

            if dsii + dsjj <= dsij + dsji:
                points += 1
                # si_idx = get_idx_in_list(s_i.tolist())
                # sj_idx = get_idx_in_list(s_j.tolist())
                # index_pairs.append([si_idx, sj_idx])
                # if f"{si_idx}_{sj_idx}" in word_pairs:
                #     word_pairs[f'{si_idx}_{sj_idx}'] += 1
                # else:
                #     word_pairs[f'{si_idx}_{sj_idx}'] = 1

            total_points += 1

    grid = np.zeros((16, 16))
    # for pair in index_pairs:
    # row, col = pair
    # # print(pair)
    # grid[row, col] += 1
    # grid[col, row] += 1

    # Next, for each word pair in the 2v2 test, increament that cell by 1. Have to make sure that the matrices are symmetric.
    # But first, you need to find out the word pair. One way is to store the word
    # pairs in an array; in other words, store the index pairs.

    gcf = None  # plot_grid(grid)
    return points, total_points, points / total_points, gcf, grid


def w2v_within_animacy_2v2(y_test, preds):
    """
    In this function we will compare only word pairs for within groups.
    """
    points = 0
    total_points = 0
    mid_idx = len(y_test) // 2

    # First for the animate words.
    for i in range(mid_idx):
        s_i = y_test[i]
        s_i_pred = preds[i]
        for j in range(i + 1, mid_idx):
            s_j = y_test[j]
            s_j_pred = preds[j]

            dsii = cosine(s_i, s_i_pred)
            dsjj = cosine(s_j, s_j_pred)
            dsij = cosine(s_i, s_j_pred)
            dsji = cosine(s_j, s_i_pred)

            if dsii + dsjj <= dsij + dsji:
                points += 1

            total_points += 1

    # Now for the inanimate words.
    for i in range(mid_idx, len(y_test)):
        s_i = y_test[i]
        s_i_pred = preds[i]
        for j in range(i + 1, len(y_test)):
            s_j = y_test[j]
            s_j_pred = preds[j]

            dsii = cosine(s_i, s_i_pred)
            dsjj = cosine(s_j, s_j_pred)
            dsij = cosine(s_i, s_j_pred)
            dsji = cosine(s_j, s_i_pred)

            if dsii + dsjj <= dsij + dsji:
                points += 1

            total_points += 1

    grid = np.zeros((16, 16))
    # for pair in index_pairs:
    # row, col = pair
    # # print(pair)
    # grid[row, col] += 1
    # grid[col, row] += 1

    # Next, for each word pair in the 2v2 test, increament that cell by 1. Have to make sure that the matrices are symmetric.
    # But first, you need to find out the word pair. One way is to store the word
    # pairs in an array; in other words, store the index pairs.

    gcf = None  # plot_grid(grid)
    return points, total_points, points / total_points, gcf, grid


def get_phoneme_idxs(word, first_or_second):
    key = labels_mapping[word]
    if first_or_second == 1:
        phoneme = first_phonemes_mapping[key]
    else:
        phoneme = stimuli_to_second_ipa_mapping[key]
    return phoneme


def extended_2v2_phonemes(y_test, preds, words, first_or_second = 1):
    """
    This test is a version of the two_vs_two test where the embeddings for the same phonemes are not compared.
    There are two additions to this function over the previous two_vs_two test.
    1. The grid figures will be symmetric now.
    2. The 16 samples in the test set will now be extended to 16C2=120 samples.
    """
    points = 0
    total_points = 0
    # diff = []
    # sum_ii_jj = []
    # sum_ij_ji = []
    # word_pairs = dict()
    # index_pairs = []
    for i in range(len(words) - 1):
        ph1 = get_phoneme_idxs(words[i], first_or_second)
        for j in range(i+1, len(words)):
            ph2 = get_phoneme_idxs(words[j], first_or_second)
            if ph1 == ph2:
                continue
            s_i = y_test[i]
            s_j = y_test[j]
            s_i_pred = preds[i]
            s_j_pred = preds[j]

            dsii = cosine(s_i, s_i_pred)
            dsjj = cosine(s_j, s_j_pred)

            dsij = cosine(s_i, s_j_pred)
            dsji = cosine(s_j, s_i_pred)
            # print("dsii: ", dsii)
            # print("dsii abs: ", np.abs(dsii[0][0]))
            # print("dsij: ", dsij)
            # print("dsji: ", dsji)
            # # print("Addition", dsii+dsjj)
            # sum_ii_jj.append((dsii + dsjj))
            # sum_ij_ji.append((dsij + dsji))
            # diff.append((dsii + dsjj) - (dsij + dsji))
            if dsii + dsjj <= dsij + dsji:
                points += 1
                # si_idx = get_idx_in_list(s_i.tolist())
                # sj_idx = get_idx_in_list(s_j.tolist())
                # index_pairs.append([si_idx, sj_idx])
                # if f"{si_idx}_{sj_idx}" in word_pairs:
                #     word_pairs[f'{si_idx}_{sj_idx}'] += 1
                # else:
                #     word_pairs[f'{si_idx}_{sj_idx}'] = 1

            total_points += 1

    grid = np.zeros((16, 16))
    # for pair in index_pairs:
    #     row, col = pair
    #     # print(pair)
    #     grid[row, col] += 1
    #     grid[col, row] += 1

    # Next, for each word pair in the 2v2 test, increament that cell by 1. Have to make sure that the matrices are symmetric.
    # But first, you need to find out the word pair. One way is to store the word
    # pairs in an array; in other words, store the index pairs.

    gcf = None  # plot_grid(grid)
    return points, total_points, points / total_points, gcf, grid


def extended_2v2_perm(y_test, preds):
    """
    There are two additions to this function over the previous two_vs_two test.
    1. The grid figures will be symmetric now.
    2. The 16 samples in the test set will now be extended to 16C2=120 samples.
    """
    points = 0
    total_points = 0
    diff = []
    sum_ii_jj = []
    sum_ij_ji = []
    x_length = [_ for _ in range(preds.shape[0] - 1)]
    word_pairs = dict()
    index_pairs = []
    for i in range(preds.shape[0] - 1):
        s_i = y_test[i]
        s_i_pred = preds[i]
        for j in range(i + 1, preds.shape[0]):
            s_j = y_test[j]
            s_j_pred = preds[j]

            si_idx = get_idx_in_list(s_i.tolist())
            sj_idx = get_idx_in_list(s_j.tolist())
            if si_idx != sj_idx:  # if the pairs of words in the 2v2 test are different.
                dsii = cosine(s_i, s_i_pred)
                dsjj = cosine(s_j, s_j_pred)
                dsij = cosine(s_i, s_j_pred)
                dsji = cosine(s_j, s_i_pred)

                sum_ii_jj.append((dsii + dsjj))
                sum_ij_ji.append((dsij + dsji))
                diff.append((dsii + dsjj) - (dsij + dsji))

                if dsii + dsjj <= dsij + dsji:
                    points += 1
                    index_pairs.append([si_idx, sj_idx])
                    if f"{si_idx}_{sj_idx}" in word_pairs:
                        word_pairs[f'{si_idx}_{sj_idx}'] += 1
                    else:
                        word_pairs[f'{si_idx}_{sj_idx}'] = 1

                total_points += 1

    grid = np.zeros((16, 16))
    for pair in index_pairs:
        row, col = pair
        # print(pair)
        grid[row, col] += 1
        grid[col, row] += 1

    # Next, for each word pair in the 2v2 test, increament that cell by 1. Have to make sure that the matrices are symmetric.
    # But first, you need to find out the word pair. One way is to store the word
    # pairs in an array; in other words, store the index pairs.

    gcf = None  # plot_grid(grid)
    return points, total_points, points / total_points, gcf, grid


def divide_by_labels(data):
    """
    :param data is an ndarray containing the values to be separated into groups.
    :param labels is an ndarray which contains the values using which the data will be grouped.
    :returns grouped data, and grouped labels.
    """
    temp_data = []
    temp_labels = []
    for i in range(16):
        t = data.loc[data.label == i]
        temp_data.append(t.iloc[:, :18000])
        temp_labels.append(t.iloc[:, 18000])

    return temp_data, temp_labels


def random_subgroup(data, labels, factor):
    group_factor = factor
    all_grouped_data = []
    all_grouped_labels = []
    for i in range(len(data)):
        curr_data = np.array(data[i])
        curr_labels = np.array(labels[i])
        shuff_idxs = np.random.permutation(len(curr_data))
        # print(shuff_idxs)
        s_curr_data = curr_data[shuff_idxs]
        s_curr_labels = curr_labels[shuff_idxs]
        grouped_data = [s_curr_data[i:i + group_factor] for i in range(0, len(s_curr_data), group_factor)]
        grouped_labels = [s_curr_labels[i:i + group_factor] for i in range(0, len(s_curr_labels), group_factor)]
        all_grouped_data.append(grouped_data)
        all_grouped_labels.append(grouped_labels)
    return all_grouped_data, all_grouped_labels


def average_grouped_data(data, labels):
    meaned_data = []
    meaned_labels = []
    for i in range(len(data)):
        curr_data_set = data[i]
        curr_label_set = labels[i]
        temp_meaned_data = []
        temp_meaned_labels = []
        for j in range(len(curr_data_set)):
            # First subgroup of the group.
            # Average the values
            avg = np.array(curr_data_set[j]).mean(axis=0)
            temp_meaned_data.append(avg)
            temp_meaned_labels.append(curr_label_set[j][0])
        meaned_data.append(temp_meaned_data)
        meaned_labels.append(temp_meaned_labels)
    data_res = []
    labels_res = []
    for l in meaned_data:
        for m in l:
            data_res.append(m.tolist())
    for l in meaned_labels:
        for m in l:
            labels_res.append(m)

    return data_res, labels_res, meaned_labels


def get_w2v_embeds(labels):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        'G:\jw_lab\jwlab_eeg\\regression\GoogleNews-vectors-negative300.bin.gz', binary=True)
    w2v_label_embeds = {}
    for key in labels_mapping:
        w2v_label_embeds[key] = model[labels_mapping[key]]
    all_embeds = []
    all_embeds.append(w2v_label_embeds)
    savez_compressed('G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\embeds_with_label_dict.npz', all_embeds)
    # for label in labels:
    #     all_embeds.append(w2v_label_embeds[int(label)])
    # return all_embeds


def get_w2v_embeds_from_dict(labels):
    embeds_with_labels_dict_loaded = load(embeds_with_label_path, allow_pickle=True)
    embeds_with_labels_dict = embeds_with_labels_dict_loaded['arr_0']
    embeds_with_labels_dict = embeds_with_labels_dict[0]

    w2v_labels = []
    for label in labels:
        w2v_labels.append(embeds_with_labels_dict[int(label)])
    w2v_labels = np.array(w2v_labels)

    return w2v_labels


def get_phoneme_onehots(labels):
    ph_embeds_npz = load(ph_first_one_hots_path)
    ph_embeds_loaded = ph_embeds_npz['arr_0']

    ph_labels = []
    for label in labels:
        ph_labels.append(ph_embeds_loaded[int(label)])
    ph_labels = np.array(ph_labels)

    return ph_labels


def get_phoneme_classes(labels):
    ph_classes_npz = load(ph_second_classes_path)  # Change path to second classes.
    ph_classes_loaded = ph_classes_npz['arr_0']

    ph_classes = []
    for label in labels:
        ph_classes.append(ph_classes_loaded[int(label)])
    ph_classes = np.array(ph_classes)

    return ph_classes


# def two_vs_two_one_hot(y_test, preds):


# def convert_to_one_hots(outputs):
#     preds = np.copy(outputs)
#     # Go over each of the preds and then set the value at position of max real value to 1 and others to zero.
#     for p in preds:


def get_sim_agg_first_embeds(labels):
    ph_sim_npz = load(sim_agg_first_embeds_path, allow_pickle=True)
    ph_sim_embeds_loaded = ph_sim_npz['arr_0'][0]

    ph_sim_embeddings = []
    for label in labels:
        ph_sim_embeddings.append(ph_sim_embeds_loaded[int(label)])

    ph_sim_embeddings = np.array(ph_sim_embeddings)
    return ph_sim_embeddings


def get_sim_agg_second_embeds(labels):
    ph_sim_npz = load(sim_agg_second_embeds_path, allow_pickle=True)
    ph_sim_embeds_loaded = ph_sim_npz['arr_0'][0]

    ph_sim_embeddings = []
    for label in labels:
        ph_sim_embeddings.append(ph_sim_embeds_loaded[int(label)])

    ph_sim_embeddings = np.array(ph_sim_embeddings)
    return ph_sim_embeddings


# Don't use this function anymore.
def remove_data(X, y):
    missing_second_phonemes = [3, 6, 10, 12]
    rmv_idxs_list = []
    for k in range(len(y)):
        if y[k] in missing_second_phonemes:
            rmv_idxs_list.append(k)

    # Now remove the corresponding data from X and y.
    X_temp = X.drop(X.index[rmv_idxs_list], axis=0)
    y_temp = np.delete(y, rmv_idxs_list, axis=0)

    return X_temp, y_temp
