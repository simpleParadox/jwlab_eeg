'''
    File name: rsa_helper.py
    Author: Wenxuan Guo
    Date created: 7/27/2020
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from scipy.stats import pearsonr

def eeg_filter_by_group(data, group, word):
    """
    Filter out data for different months - for single word. For example. 9 month olds - word 0, 12 month olds word 1, etc.
    Args:
        data: the eeg data
        group: the age group
        word: the word label
    Return:
        the dataset that contains the subset of eeg for an age group for a specific word.
    """
    if group == 9:
        data = data.iloc[1008:]
        data = data[data['label']==float(word)]#.iloc[:, :18000].values
    elif group == 12:
        data = data[:1008]
        data = data[data['label'] == float(word)]#.iloc[:,:18000].values
    else:
        data = data[data['label'] == float(word)]#.iloc[:, 18000].values

    # Now average the readings for each subject for that word.
    participants = [i for i in range(min(data['participant']), max(data['participant']))]
    eeg_means = []
    for part in participants:
        eeg_data = data[data['participant']==part].iloc[:, :18000].values
        eeg_means.append(np.mean(eeg_data, axis=0))
    eeg_means = np.array(eeg_means)
    return eeg_means, participants, 'all 9 months', word


def eeg_filter_subject(data, subject, word):
    """
    Returns the data for one subject only for a specific word
    Args:
        data: the whole dataset
        subject: the participant number
        word: the word label
    Returns:
        The data for a single participant for a specific word.
    """
    data = data[data['participant']==subject]
    data = data[data['label']==float(word)].iloc[:,:18000]#.values

    return data, subject, word


def eeg_filter_subject_all_words(data, ps):
    """
    Filter out all the words averaged together for a single participants
    """
    data = data[data['participant'] == ps]
    labels = [i for i in range(16)]
    ps_data = []
    for word in labels:
        words = data[data['label'] == float(word)].iloc[:, :18000].values
        ps_data.append(np.nanmean(words, axis=0))

    ps_data = np.array(ps_data)
    return ps_data, ps, 'all words'

def PCA_selection(X, percent_variance = .9):
    pca = PCA(svd_solver='auto')
    pca.fit(X)
    csum_variance = np.cumsum(pca.explained_variance_ratio_)
    # find the first q vectors that inform 90% of the variance
    q = len(csum_variance[csum_variance < percent_variance]) + 1

    pca = PCA(n_components=200, svd_solver='auto')
    pca.fit(X)
    score = pca.transform(X)

    return score

def PFA_selection(X, n_features, percent_variance = .9):
    """
    Args:
        X: input matrix of shape (nsamples, nfeatures)
        n_features: the number of features used for PFA
        percent_variance: lowest percentage of the variance explained by the first q components of PCA; default = 90%
    """
    q_evectors = PCA_selection(X, percent_variance=percent_variance)

    # Choose the subspace dimension q and construct the matrix
    # the transformed matrix A_q
    A_q = q_evectors.T
    # n_features should be higher than q
    kmeans = KMeans(n_clusters=n_features).fit(A_q)
    # For each cluster, only the vector closest to the center of cluster is retained,
    clusters = kmeans.predict(A_q)
    # Coordinates of cluster centers
    cluster_centers = kmeans.cluster_centers_

    dists = dict(list)
    for i, c in enumerate(clusters):
        dist = euclidean_distances(
            [A_q[i, :]], [cluster_centers[c, :]])[0][0]
        dists[c].append((i, dist))

    # the column indices of the kept features
    indices = [sorted(f, key=lambda x: x[1])[0][0]
                                for f in dists.values()]
    # the transformed matrix
    features = X[:, indices]

    return indices, features

def eeg_rdm_dist_corr(score, type='pearson'):
    """
    calculates an RDM from the fmri voxels using correlation distance
    Args:
        X: the original data
        evectors: the first q principal components
    """
    # number of stimuli
    num_stimuli = score.shape[0]
    # initialize RDM
    RDM = np.ones((num_stimuli, num_stimuli))
    # compute correlation distance between vectors
    for i in range(num_stimuli):
        for j in range(num_stimuli):
            if type == 'pearson':
                RDM[i][j] = 1 - pearsonr(score[i], score[j])[0]
            else:
                RDM[i][j] = distance.correlation(score[i], score[j])
    
    return RDM



def word_RDM(model, label_list):
    # number of stimuli
    num_stimuli = len(label_list)
    # initialize RDM
    RDM = np.ones((num_stimuli, num_stimuli))
    # compute cosine similarity between words
    for i in range(num_stimuli):
        for j in range(num_stimuli):
            try:
                similarity = model.wv.similarity(label_list[i], label_list[j])
                RDM[i][j] = 1 - similarity # dissimilarity
            except:
                pass
    
    return RDM

def RDM_vis(RDM, subject, word):
    plt.figure()
    plt.matshow(RDM)
    plt.colorbar()
    plt.title('RDM PS: '+ str(subject) + ' / word: ' + str(word))
    plt.show()