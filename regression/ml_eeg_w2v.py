
import sklearn
import pandas as pd
import numpy as np
import pickle
import gensim
from numpy import savez_compressed
from numpy import load
import platform
import time


from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
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


os = platform.system()
readys_path = None
if os =='Windows':
    readys_path = "Z:\\Jenn\\ml_df_readys.pkl"
elif os=='Linux':
    readys_path = os.getcwd() + "/data/ml_df_readys.pkl"

# with open(pkl_path, 'rb') as f:
f = open(readys_path, 'rb')
readys_data = pickle.load(f)
f.close()



eeg_features = readys_data.iloc[:,:18000].values
w2v_path = None
if os =='Windows':
    w2v_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\all_w2v_embeds.npz"
elif os=='Linux':
    w2v_path = os.getcwd() + "/w2v_embeds/all_w2v_embeds.npz"
w2v_embeds_loaded = load(w2v_path)
w2v_embeds = w2v_embeds_loaded['arr_0']

print("Data Loaded")
print("Readys shape: ", eeg_features.shape)
print("w2v shape: ", w2v_embeds.shape)

def monte_carlo_2v2():
    start = time.time()
    print("Monte-Carlo CV")
    # Split into training and testing data
    parameters_ridge = {'alpha': [10000000, 100000000, 1000000000]} #0.01]}#, 0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000,
    parameters_dt = {'random_state': [None]}

    ridge = DecisionTreeRegressor()
    clf = GridSearchCV(ridge, param_grid=parameters_dt, scoring='neg_mean_squared_error',
                       refit=True, cv=2, verbose=5)


    rs = ShuffleSplit(n_splits=2, train_size=0.90)
    all_data_indices = [i for i in range(len(w2v_embeds))]
    f = 1
    score_with_alpha = {}
    cosine_scores = []
    for train_index, test_index in rs.split(all_data_indices):
        print("Shuffle Split fold: ", f)
        X_train, X_test = eeg_features[train_index], eeg_features[test_index]
        y_train, y_test = w2v_embeds[train_index], w2v_embeds[test_index]

        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print("Preds", preds.shape)
        print("y_test:", y_test.shape)
        f += 1
        points = 0
        total_points = 0
        for i in range(preds.shape[0]-1):
            s_i = y_test[i]
            s_j = y_test[i+1]
            s_i_pred = preds[i]
            s_j_pred = preds[i+1]
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
        print("Points: ", points)
        print("Total points: ", total_points)
        acc = points / total_points
        cosine_scores.append(acc)
        print(acc)
    score_with_alpha[str(clf.best_params_)] = np.average(np.array(cosine_scores), axis=0)
    print("All scores: ", score_with_alpha)
    stop = time.time()
    print("Total time: ", stop - start)

# monte_carlo_2v2()


