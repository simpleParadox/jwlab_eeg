#!/usr/bin/env python
# coding: utf-8

# In[1]:


from curses import meta
from pydoc import HTMLRepr
import pandas as pd
import numpy as np
import setup_jwlab

import sys
import argparse
import wandb
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')


parser = argparse.ArgumentParser(description='Run the decoding analysis.')

parser.add_argument('--seed', type=int, default=0, help='Random seed for the analysis')
parser.add_argument('--graph_file_name', type=str, default='gpt2-xl', help='Graph file name')
parser.add_argument('--model_name', type=str, default=None, help='Model name')
parser.add_argument('--layer', type=int, default=1, help='Layer number')
parser.add_argument('--use_randomized_label', default=False, action='store_true', help='Use randomized labels')
parser.add_argument('--age_group', type=int, default=9, help='Age group')
parser.add_argument('--iterations', type=int, default=50, help='Number of sampling iterations to run')
parser.add_argument('--fixed_seed', default=False, action='store_true', help='Whether to fix seeds for replicability.')
parser.add_argument('--embedding_type', default='w2v', type=str, help='Embedding type w2v or ph when --model_name is None. In case --model_name is provided, this will be ignored and --model_name will take precedence.')
parser.add_argument('--svd_vectors', default=False, action='store_true', help='Whether to use SVD vectors for the embeddings')
parser.add_argument('--wandb_mode', type=str, default='online', help='Wandb mode: online or offline')
parser.add_argument('--iteration_range', type=int, nargs=2, metavar=('start', 'end'), default=None, help='Range of iterations to run')


parsed_args = parser.parse_args()
wandb.login()


print("Running job with args: ", parsed_args)
seed = parsed_args.seed
graph_file_name = parsed_args.graph_file_name
model_name = parsed_args.model_name
layer = parsed_args.layer
if parsed_args.iteration_range:
    iterations = range(parsed_args.iteration_range[0], parsed_args.iteration_range[1])
else:
    iterations = parsed_args.iterations
print("Iterations: ", iterations)
# Append the graph_file_name with the layer value.
graph_file_name = graph_file_name + f'_{layer}' + f'_iterations_{iterations}'
if parsed_args.svd_vectors:
    graph_file_name = graph_file_name + '_svd_vectors' # Denote the vectors as svd vectors.
print("Graph file name: ", graph_file_name)
use_randomized_label = parsed_args.use_randomized_label
if use_randomized_label:
    graph_file_name = graph_file_name + '_randomized_labels'
print("Randomized labels: ", use_randomized_label)
age_group = parsed_args.age_group
fixed_seed = parsed_args.fixed_seed
embedding_type = parsed_args.embedding_type


# Log all the arguments to wandb by first putting them in a dictionary.
run = wandb.init(project='jwlab-eeg', entity='simpleparadox', mode=parsed_args.wandb_mode,
                       config=parsed_args)
#Update the graph_file_name to the wandb config.
wandb.config.update({'graph_file_name': graph_file_name}, allow_val_change=True)


sys.path.insert(1, '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/classification/code')
from jwlab.constants import cleaned_data_filepath
from jwlab.cluster_analysis_perm import cluster_analysis_procedure
from jwlab.ml_prep_perm import prep_ml, slide_df, init, load_ml_data, get_bad_trials, map_participants,average_trials_and_participants
from jwlab.bad_trials import get_bad_trials, get_left_trial_each_word


from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, RepeatedKFold
from scipy import stats
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


# In[3]:




# Argument 1: 9 or 11 (month olds)
# Argument 2: Boolean, True to randomize the labels, False otherwise
# Argument 3: averaging, could be: no_averaging, average_trials, average_trials_and_participants, permutation
# Argument 4: sliding_window_config[start_time, end_time, window_lengths[], step_length]
# Argument 5: cross_val_config[num_fold, num_fold iterations, number of sample iterations]

#cluster_analysis_procedure(9, False, "permutation", [-200, 1000, [10, 20, 40, 60], 10], [3, 5])


# In[ ]:



# NOTE: If you set useRandomizedLabel = True and set type='simple', it will run the null_distribution / permutation test. But you have to run it 100 times/jobs.
result = cluster_analysis_procedure(age_group, use_randomized_label, 
                                    "average_trials_and_participants",
                                    [-200, 1000, [100], 10], 
                                    [5, 4, iterations], 
                                    type_exp='simple', # 'permutation' or 'simple'
                                    animacy=False, 
                                    no_animacy_avg=False, 
                                    do_eeg_pca=False, 
                                    do_sliding_window=False,
                                    model_name=model_name,
                                    layer=layer,
                                    graph_file_name=graph_file_name,
                                    fixed_seed=fixed_seed,
                                    embedding_type=embedding_type,
                                    wandb_object=wandb) # Max layer must be 36 for gpt2-large and 48 for gpt2-xl (the numbers are 'indices' of the layer).


run.finish()
# group_num = 0 #int(sys.argv[2])

# start_wind = 0 #int(sys.argv[4])
# end_wind = 300 #int(sys.argv[5])


# result = cluster_analysis_procedure(age_group, False, "average_trials_and_participants", [start_wind, end_wind, [end_wind - start_wind], 10], [5, 4, 50], type='simple', animacy=False, no_animacy_avg=False, do_eeg_pca=False, 
#                                     do_sliding_window=False, ch_group=True, group_num=group_num)
# result = cluster_analysis_procedure(9, False, "average_trials_and_participants", [-200, 1000, [100], 10], [5, 4, 70], type='simple', residual=True, child_residual=False)

# result = cluster_analysis_procedure(12, False, "tgm", [-200, 1000, [100], 10], [5, 4, 50], type='simple', seed=seed, corr=False, target_pca=False)
# result = cluster_analysis_procedure(12, False, "across", [-200, 1000, [100], 10], [5, 4, 10], type='simple', seed=seed, corr=False, target_pca=False, animacy=False)

# In[25]:


# import statistics
# negsum = 0
# negsum += stats.ttest_1samp(result[0][15], .5).statistic
# negsum += stats.ttest_1samp(result[0][16], .5).statistic
# negsum += stats.ttest_1samp(result[0][17], .5).statistic
# negsum += stats.ttest_1samp(result[0][18], .5).statistic
# negsum += stats.ttest_1samp(result[0][19], .5).statistic


# # In[26]:


# negsum


# # In[6]:


# results = cluster_analysis_procedure(9, False, "average_trials_and_participants", [-200, 1000, [100], 10], [1, 1, 20])


# # In[ ]:


# num_win= 120

# pvalues_pos = []
# pvalues_neg = []
# tvalues_pos = []
# tvalues_neg = []
# for i in range(len(results)):
#     for j in range(num_win):
#         # change the second argument below for comparison
#         istat = stats.ttest_1samp(results[i][j], .5)
#         pvalues_pos += [istat.pvalue] if istat.statistic > 0 else [1]
#         pvalues_neg += [istat.pvalue] if istat.statistic < 0 else [1]
#         # removed just so that we can get the negative value from the pre window
#         tvalues_pos += [istat.statistic] if istat.statistic > 0 else [0]
#         tvalues_neg += [istat.statistic] if istat.statistic < 0 else [0]


# # In[ ]:





# # In[ ]:





# # In[2]:


# # For null distribution
# # MAKE SURE YOU TURN OFF PRINT FUNCTION

# itr = 50 
# arrTmass = []

# for i in range(itr):
#     tmass = cluster_analysis_procedure(11, True, "permutation", [-200, 1000, [10], 10], [3, 15, 20])
#     arrTmass.append(round(tmass, 4))
#     print(i)
# print(arrTmass)


# # In[ ]:




# plt.hist(arrTmass, bins = 20)
# plt.show()


# In[ ]:





# In[ ]:




