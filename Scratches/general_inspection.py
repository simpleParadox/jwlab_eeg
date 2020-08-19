# Scratch python script for inspecting new files.
import numpy as np
from scipy.io import loadmat
import pandas as pd
import pickle
import os
import platform
os_name = platform.system()

if os_name == 'Windows':
    from regression.functions import average_trials, average_trials_and_participants, labels_mapping, two_vs_two, \
        test_model, test_model_permute, two_vs_two_test, divide_by_labels, random_subgroup, average_grouped_data, \
        get_w2v_embeds, get_w2v_embeds_from_dict
    # from regression.other_exps import labels_mapping_mod_ratings
else:
    from functions import average_trials, average_trials_and_participants, labels_mapping, two_vs_two, test_model, \
        test_model_permute, two_vs_two_test, divide_by_labels, random_subgroup, average_grouped_data, get_w2v_embeds, \
        get_w2v_embeds_from_dict
    from functions import labels_mapping_mod_ratings





readys_path = None
avg_readys_path = None
if os_name == 'Windows':
    # readys_path = "Z:\\Jenn\\ml_df_readys.pkl"
    readys_path = "G:\\jw_lab\\jwlab_eeg\\regression\\data\\ml_df_readys.pkl"
    avg_readys_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_data_readys.pkl"
    avg_trials_and_ps_9m_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_and_ps_9m.pkl"
    avg_trials_and_ps_13m_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_and_ps_13m.pkl"
    avg_trials_and_ps_9and13_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_and_ps_9and13.pkl"
    bag_of_features = "G:\\jw_lab\\jwlab_eeg\\regression\data\\bagOfFeatures (1).mat"
elif os_name == 'Linux':
    readys_path = os.getcwd() + "/regression/data/ml_df_readys.pkl"
    avg_readys_path = os.getcwd() + "/regression/data/avg_trials_data_readys.pkl"
    avg_trials_and_ps_9m_path = os.getcwd() + "/regression/data/avg_trials_and_ps_9m.pkl"
    avg_trials_and_ps_13m_path = os.getcwd() + "/regression/data/avg_trials_and_ps_13m.pkl"
    avg_trials_and_ps_9and13_path = os.getcwd() + "/regression/data/avg_trials_and_ps_9and13.pkl"
    bag_of_features = os.getcwd() + "/regression/data/bagOfFeatures (1).mat"

data = loadmat(bag_of_features)
# In the .mat file, there are three ndarrays -> column labels, the actual data, and the word labels.
# The columns labels are the questions which were asked to the participants; each cell value is a rating from 1-5.

labels_mapping_mod_ratings = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'rabbit',
                              4: 'cat', 5: 'dog', 6: 'duck',
                              8: 'banana', 9: 'bottle', 10: 'cookie',
                              11: 'biscuit', 12: 'cup', 13: 'juice',
                              14: 'milk', 15: 'spoon'}

# Let's do some inspection of the ratings.
questions = []
for q in data['featureLabels']:
    questions.append(q[0][0])
    # print(q[0][0])

questions = np.array(questions)
# np.savetxt('temp.csv', questions, delimiter=',', )

# pd.DataFrame(questions).to_csv('temp.csv')
filtered_col_idxs = [0,13,14,21,25,26,27,29,30,31,34,45,91,92,119,121,122]

nouns = [temp[0][0].lower() for temp in data['nouns']]
f = open(readys_path, 'rb')
readys_data = pickle.load(f)
f.close()
labels = readys_data.iloc[:, 18000].values
# # First note the indices where the the label is 7 -> word 'mom
filtered_idxs = []  # Contains indices from readys_data without the label '7'.
for l in range(len(labels)):
    if labels[l] != 7:
        filtered_idxs.append(l)
filtered_readys_data = readys_data.iloc[filtered_idxs,:18000].values # Store this
readys_ratings = []  # Store this. The eeg data and corresponding ratings.
for j in filtered_idxs:
    lab_idx = labels[j]  # This is in agreement to the labels_mod_mapping.
    # print(lab_idx)
    word = labels_mapping_mod_ratings[lab_idx]
    bof_rating_index = nouns.index(word)
    rating = data['features'][bof_rating_index]
    readys_ratings.append(rating)
readys_ratings = np.array(readys_ratings)

# Now selecting only the subset of questions from the ratings.

readys_ratings_reduced = readys_ratings[:,filtered_col_idxs]

scores = []
for _ in range(10):
    score = test_model(filtered_readys_data, readys_ratings_reduced)
    scores.append(score)

print(np.mean(scores))