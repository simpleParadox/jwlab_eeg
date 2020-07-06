import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
import pandas as pd
import pickle
import random
import gensim
from numpy import load, savez_compressed
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

word_list = ["baby", "BAD_STRING", "bird", "BAD_STRING", "cat", "dog", "duck", "mommy",
             "banana", "bottle", "cookie", "cracker", "BAD_STRING", "juice", "milk", "BAD_STRING"]

labels_mapping = {0:'baby', 1:'bear', 2:'bird', 3: 'bunny',
                      4:'cat', 5 : 'dog', 6: 'duck', 7: 'mom',
                      8: 'banana', 9: 'bottle', 10: 'cookie',
                      11: 'cracker', 12: 'cup', 13: 'juice',
                      14: 'milk', 15: 'spoon'}
def test_model_permute(X, y):
    random.shuffle(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, shuffle=True)
    model = Ridge()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    a,b,c = two_vs_two(y_test, preds)
    print(c)

def test_model(X, y):
    # random.shuffle(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True)
    print(X_train.shape)
    print(y_train.shape)
    model = Ridge()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    a,b,c = two_vs_two(y_test, preds)
    print(c)
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

def two_vs_two(y_test, preds):
    # print("Ytest", y_test[0])
    # print("Preds",preds[0])
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
    model = gensim.models.KeyedVectors.load_word2vec_format('G:\jw_lab\jwlab_eeg\\regression\GoogleNews-vectors-negative300.bin.gz', binary=True)
    w2v_label_embeds = {}
    for key in labels_mapping:
        w2v_label_embeds[key] = model[labels_mapping[key]]
    all_embeds = []
    all_embeds.append(w2v_label_embeds)
    savez_compressed('G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\embeds_with_label_dict.npz', all_embeds)
    # for label in labels:
    #     all_embeds.append(w2v_label_embeds[int(label)])
    # return all_embeds
