import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
import pandas as pd
import pickle

word_list = ["baby", "BAD_STRING", "bird", "BAD_STRING", "cat", "dog", "duck", "mommy",
             "banana", "bottle", "cookie", "cracker", "BAD_STRING", "juice", "milk", "BAD_STRING"]

labels_mapping = {0:'baby', 1:'bear', 2:'bird', 3: 'bunny',
                      4:'cat', 5 : 'dog', 6: 'duck', 7: 'mom',
                      8: 'banana', 9: 'bottle', 10: 'cookie',
                      11: 'cracker', 12: 'cup', 13: 'juice',
                      14: 'milk', 15: 'spoon'}

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
            means = df_data[np.logical_and(df.participant == p, df.label == w)].values.mean(axis=0) if df_data[np.logical_and(df.participant == p, df.label == w)].size != 0 else 0
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
            means = df_data[np.logical_and(n_df.participant == p, n_df.label == w)].values.mean(axis=0) if df_data[np.logical_and(n_df.participant == p, n_df.label == w)].size != 0 else 0
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
        mean = count / 13  #len(participants)
        new_data[w, :] = mean
        new_y[w] = -1 if np.isnan(mean).any() else w
    new_data = new_data[new_y != -1, :]
    new_y = new_y[new_y != -1]
    new_data_df = pd.DataFrame(new_data)
    new_data_df.to_pickle('G:\\jw_lab\\jwlab_eeg\\regression\\data\\avg_trials_and_ps_13m.pkl')
    return new_data, new_y, np.ones(new_y.shape[0]) * -1, np.copy(new_y)


def two_vs_two(y_test, preds):
    points = 0
    total_points = 0
    for i in range(preds.shape[0] - 1):
        s_i = y_test[i]
        s_j = y_test[i + 1]
        s_i_pred = preds[i]
        s_j_pred = preds[i + 1]
        # print("S_i", s_i)
        # print('s_i_pred', s_i_pred)
        dsii = cosine_similarity([s_i], [s_i_pred])
        dsjj = cosine_similarity([s_j], [s_j_pred])
        dsij = cosine_similarity([s_i], [s_j_pred])
        dsji = cosine_similarity([s_j], [s_i_pred])
        # print("dsii: ", dsii)
        # print("dsjj: ", dsjj)
        # print("dsij: ", dsij)
        # print("dsji: ", dsji)
        if (dsii + dsjj) >= (dsij + dsji):
            points += 1
        total_points += 1
    return points, total_points, points / total_points
