import pandas as pd
import numpy as np
import random
from scipy import stats
import more_itertools as mit
import matplotlib as mpl

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def monte_carlo_animacy_from_vectors():
    preds = np.load('/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/classification/code/jwlab/w2v_preds/12_1m_w2v_pred_vecs_from_eeg.npz', allow_pickle=True)
    preds = preds['arr_0'].tolist()

    mod_d = {}
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            d = np.vstack(preds[i][j])
            mod_d[j] = d
    
    ld = dict()
    ld[0] = mod_d



    global_acc = {}
    # for i in range(len(preds[0])):
    #     global_acc[i] = []

    y_embed_labels = [i for i in range(0, 16)]
    scoring = 'neg_mean_squared_error'
    lr_params = {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    y = np.array([0 if t < 8 else 1 for t in y_embed_labels])
    y = np.array(y.tolist() * len(preds[0][0]))

    for i in range(len(ld)):
        for j in range(len(ld[i])):
            # 'j' goes from 0-110 (total 111).
            wind_accs = []
            x = ld[i][j]
            sf = ShuffleSplit(50, test_size=0.20)
            accs = []
            for train_idx, test_idx in sf.split(x):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = LogisticRegression()
                cv = GridSearchCV(model, param_grid=lr_params, scoring=scoring, cv=5, n_jobs=-1)
                cv.fit(x_train, y_train)
                y_preds = cv.predict(x_test)

                # Now compare the preds and true_values
                acc = (y_preds == y_test).sum() / len(y_test)
                accs.append(acc)
            global_acc[j] = accs
    
    res_d = dict()
    res_d[0] = global_acc
    print('Results: ', res_d)
    

    return res_d


res = monte_carlo_animacy_from_vectors()