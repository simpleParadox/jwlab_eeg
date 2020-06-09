import sklearn
import pandas as pd
import numpy as np
import pickle
import gensim
from numpy import savez_compressed
from numpy import load
import platform
import time
import random
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import gensim
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
# from sklearn.svm import LinearSVR
# from sklearn.svm import SVR
# from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from regression.functions import average_trials

os_name = platform.system()
readys_path = None
if os_name =='Windows':
    readys_path = "Z:\\Jenn\\ml_df_readys.pkl"
elif os_name=='Linux':
    readys_path = os.getcwd() + "/regression/data/ml_df_readys.pkl"

# with open(pkl_path, 'rb') as f:
f = open(readys_path, 'rb')
readys_data = pickle.load(f)
f.close()



eeg_features = readys_data.iloc[:,:18000].values
w2v_path = None
if os_name =='Windows':
    w2v_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\all_w2v_embeds.npz"
elif os_name=='Linux':
    w2v_path = os.getcwd() + "/regression/w2v_embeds/all_w2v_embeds.npz"
w2v_embeds_loaded = load(w2v_path)
w2v_embeds = w2v_embeds_loaded['arr_0']

print("Data Loaded")
print("Readys shape: ", eeg_features.shape)
print("w2v shape: ", w2v_embeds.shape)


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
    return points, total_points, points / total_points  # The last value is the score.

def avg_trials_model():
    pass

def create_avg_data_embeds():
    # First average the trials
    labels_mapping = {0:'baby', 1:'bear', 2:'bird', 3: 'bunny',
                      4:'cat', 5 : 'dog', 6: 'duck', 7: 'mom',
                      8: 'banana', 9: 'bottle', 10: 'cookie',
                      11: 'cracker', 12: 'cup', 13: 'juice',
                      14: 'milk', 15: 'spoon'}
    avg_trial_data, labels, participants, labels_copy = average_trials(readys_data)
    # Now create the word2vec embeddings from the labels.
    # word_labels = readys_data['label'].values
    model = gensim.models.KeyedVectors.load_word2vec_format('regression\GoogleNews-vectors-negative300.bin.gz', binary=True)
    # First obtain all the embeddings for the words in the labels mapping.
    w2v_label_embeds = {}
    # pca = PCA(n_components=15)
    for key in labels_mapping:
        w2v_label_embeds[key] = model[labels_mapping[key]]
    all_embeds = []
    for label in labels:
        all_embeds.append(w2v_label_embeds[int(label)])
    savez_compressed('regression/w2v_embeds/all_w2v_embeds_pcs_avg_trial.npz', all_embeds)
    avg_data = pd.DataFrame(avg_trial_data)



def split_ps_model():
    start = time.time()
    # Split the readys data into 9 month and 13 month olds.
    # last 13 month old index => 1007.
    t_eeg = eeg_features[:1008, :]
    n_eeg = eeg_features[1008:, :]



    t_w2v = w2v_embeds[:1008, :]
    n_w2v = w2v_embeds[1008:, :]
    # Fitting on 9 month old.
    rounds = 2
    t_scores = []
    n_scores = []
    for r in range(rounds):

        t_X_train, t_X_test, t_y_train, t_y_test = train_test_split(t_eeg, t_w2v, train_size=0.90)
        n_X_train, n_X_test, n_y_train, n_y_test = train_test_split(n_eeg, n_w2v, train_size=0.90)

        print("Fitting on the 13 month olds.")
        t_model = DecisionTreeRegressor()
        t_model.fit(t_X_train, t_y_train)
        t_preds = t_model.predict(t_X_test)
        t_points, t_total_points, t_score = two_vs_two(t_y_test, t_preds)
        t_scores.append(t_score)
        print("Score for 13 month olds - no cv, no hyper optim: ", t_score)
        print("------------------------------------------------------------")
        print("Fitting on the 9 month olds.")
        n_model = DecisionTreeRegressor()
        n_model.fit(n_X_train, n_y_train)
        n_preds = n_model.predict(n_X_test)
        n_points, n_total_points, n_score = two_vs_two(n_y_test, n_preds)
        n_scores.append(n_score)
        print("Score for 9 month olds - no cv, no hyper optim: ", n_score)
    print("Average score for thirteen month old: ", np.average(t_scores))
    print("Average score for nice month old: ", np.average(n_scores))
    stop = time.time()
    print("Total time taken: ", stop - start)




def monte_carlo_2v2():
    start = time.time()
    print("Monte-Carlo CV")
    # Split into training and testing data
    parameters_ridge = {'alpha': [10000000, 100000000, 1000000000]} #0.01]}#, 0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000,
    parameters_dt = {'random_state': [None]}

    ridge = DecisionTreeRegressor()
    clf = GridSearchCV(ridge, param_grid=parameters_ridge, scoring='neg_mean_squared_error',
                       refit=True, cv=5, verbose=5)


    rs = ShuffleSplit(n_splits=10, train_size=0.90)
    all_data_indices = [i for i in range(len(w2v_embeds))]
    f = 1
    score_with_alpha = {}
    cosine_scores = []
    for train_index, test_index in rs.split(all_data_indices):
        print("Shuffle Split fold: ", f)
        X_train, X_test = eeg_features[train_index], eeg_features[test_index]
        # The following two lines are for the permutation test. Comment them out when not using the permutation test.
        random.shuffle(train_index) # For permutation test only.
        random.shuffle(test_index) # For permutation test only.
        y_train, y_test = w2v_embeds[train_index], w2v_embeds[test_index]

        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)

        ridge.fit(X_train, y_train)
        preds = ridge.predict(X_test)
        # print("Preds", preds.shape)
        # print("y_test:", y_test.shape)
        f += 1
        points, total_points, score = two_vs_two(y_test, preds)
        print("Points: ", points)
        print("Total points: ", total_points)
        acc = points / total_points
        cosine_scores.append(acc)
        print(acc)
    score_with_alpha['avg'] = np.average(np.array(cosine_scores), axis=0)
    print("All scores: ", score_with_alpha)
    stop = time.time()
    print("Total time: ", stop - start)

# monte_carlo_2v2()

# split_ps_model()

