
import sklearn
import pandas as pd
import numpy as np
import pickle
import gensim
from numpy import savez_compressed
from numpy import load
from sklearn.model_selection import train_test_split

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




readys_path = "Z:\\Jenn\\ml_df_readys.pkl"

# with open(pkl_path, 'rb') as f:
f = open(readys_path, 'rb')
readys_data = pickle.load(f)
f.close()



eeg_features = readys_data.iloc[:,:18000].values
w2v_embeds_loaded = load('w2v_embeds/all_w2v_embeds.npz')
w2v_embeds = w2v_embeds_loaded['arr_0']


# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(eeg_features, w2v_embeds, test_size=0.1)
print(X_train.shape)
print(y_train.shape)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# X_train = X_train.reshape(X_train.shape[0],-1)
# y_train = y_train.reshape(y_train.shape[0],300)
ridge = DecisionTreeRegressor()
ridge.fit(X_train, y_train)
preds = ridge.predict(X_test)

points = 0
total_points = 0
for i in range(preds.shape[0]-1):
    s_i = y_test[i]
    s_j = y_test[i+1]
    s_i_pred = preds[i]
    s_j_pred = preds[i+1]
    dsii = cosine_similarity([s_i], [s_i_pred])
    dsjj = cosine_similarity([s_j], [s_j_pred])
    dsij = cosine_similarity([s_i], [s_j_pred])
    dsji = cosine_similarity([s_j], [s_i_pred])
    if (dsii + dsjj) >= (dsij + dsji):
        points += 1
    total_points += 1
print( points / total_points)


