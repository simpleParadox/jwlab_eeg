import pandas as pd
import numpy as np
import random
from scipy import stats
import more_itertools as mit
import seaborn as sns
import matplotlib as mpl
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
from sklearn.decomposition import TruncatedSVD
# import trex as tx
# pattern = tx.compile(['baby', 'bat', 'bad'])
# hits = pattern.findall('The baby was scared by the bad bat.')
# hits = ['baby', 'bat', 'bad']
sys.path.insert(1, '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/classification/code')  ## For loading the following files.

from jwlab.ml_prep_perm import prep_ml, prep_matrices_avg
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
sys.path.insert(1, '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg')
from regression.functions import get_w2v_embeds_from_dict, two_vs_two, extended_2v2_phonemes, extended_2v2_perm, \
    get_phoneme_onehots, get_phoneme_classes, get_sim_agg_first_embeds, get_sim_agg_second_embeds, extended_2v2, w2v_across_animacy_2v2, w2v_within_animacy_2v2, \
    ph_within_animacy_2v2, ph_across_animacy_2v2, get_audio_amplitude, get_stft_of_amp, get_cbt_childes_w2v_embeds, get_all_ph_concat_embeds, \
    get_glove_embeds, get_reduced_w2v_embeds, sep_by_prev_anim


def monte_carlo_aniamcy_from_vectors():
    y_embed_labels = [i for i in range(0,16)]
    scoring = 'neg_mean_squared_error'
    lr_params = {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    y_vectors = get_w2v_embeds_from_dict(y_embed_labels)
    x = y_vectors
    y = np.array([0 if t < 8 else 1 for t in y_embed_labels])
    perm_accs = []
    for i in range(1000):
        # print(i)
        np.random.shuffle(y)
        sf = ShuffleSplit(50, test_size=0.30)
        accs = []
        for train_idx, test_idx in sf.split(x):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = LogisticRegression()
            cv = GridSearchCV(model, param_grid=lr_params, scoring=scoring, cv=6, n_jobs=-1)
            cv.fit(x_train, y_train)
            preds = cv.predict(x_test)

            # Now compare the preds and true_values
            acc = (preds == y_test).sum() / len(y_test)
            accs.append(acc)
        perm_accs.append(np.mean(acc))

    print("Accuracy: ", np.mean(perm_accs))


monte_carlo_aniamcy_from_vectors()