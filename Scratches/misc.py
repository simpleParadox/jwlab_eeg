# NOTE: This file requires a 'results' dictionary from the cluster_analysis_perm.py file.

import pandas as pd
import numpy as np
import random
from scipy import stats
import more_itertools as mit
import time
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt, cm


# The following section looks into how we can reduce the dimensions of the full Word2Vec matrix using SVD.
from gensim.models import KeyedVectors
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
model = KeyedVectors.load_word2vec_format("G:\jw_lab\jwlab_eeg\\regression\GoogleNews-vectors-negative300.bin", binary=True)

labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}



## The commented lines are old code - should not be used.
# embeds = []
# for label in labels_mapping.values():
#     embeds.append(model.wv[label])
# embeds = np.array(embeds)
#
# svd = TruncatedSVD(n_components=16, n_iter=50, random_state=42)
# embeds_svd = svd.fit_transform(embeds)
# print(svd.explained_variance_ratio_.sum())
#
# np.savez_compressed("G:\jw_lab\jwlab_eeg\\regression\w2v_embeds\pre_w2v_svd_16_components.npz", embeds_svd)
#
#
#
# pca = PCA(n_components=16, random_state=42)
# embeds_pca = pca.fit_transform(embeds)
# print(pca.explained_variance_ratio_.sum())
#
# np.savez_compressed("G:\jw_lab\jwlab_eeg\\regression\w2v_embeds\pre_w2v_pca_16_components.npz", embeds_pca)




# final_res = {}
#
# for wind in range(111):
#     final_res[wind] = []
#
# for wind in range(111):
#     for d in res:
#         perm = d[0]
#         final_res[wind].extend(perm[wind])
#
# final_res_arr = np.array(final_res)
# # Now save the 'final_res' object to disk.
# np.savez_compressed('G:\jw_lab\jwlab_eeg\Results\\12m_res_prew2v_from_eeg_null_dist_100x50iters.npz', final_res_arr)

#
# from regression.functions import extended_2v2, get_w2v_embeds_from_dict
# import numpy as np
#
# preds = np.load('G:\jw_lab\jwlab_eeg\classification\code\jwlab\w2v_preds\\12_1m_w2v_pred_vecs_from_eeg.npz', allow_pickle=True)
# preds = preds['arr_0'].tolist()
# true_preds = get_w2v_embeds_from_dict([i for i in range(0,16)])
# for i in range(len(preds)):
#     temp_results = {}
#     for j in range(len(preds[i])):
#         print(j)
#         accs = []
#         for k in range(len(preds[i][j])):
#             y_pred = preds[i][j][k]
#             res = extended_2v2(true_preds, y_pred)
#
#             accs.append(res[2])
#         temp_results[j] = accs
#
# res = {}
# res[0] = temp_results



"""
Implementing Kernel density estimation of the null_distribution results.
"""
from scipy import stats
def do_kernel_smoothing(vals, n):
    kernel = stats.gaussian_kde(vals)
    return kernel.resample(n)

#-------------------------------------------------------------------------------------
"""
This section takes all the permutation test results that are stored in a directory and then collates them into an array.
Select the block and then run it in console.
"""
import glob
import numpy as np
numpy_vars = {}
i = 0
for np_name in glob.glob('G:\jw_lab\jwlab_eeg\Results\\9m permutation results\*.np[yz]'):
    numpy_vars[i] = np.load(np_name, allow_pickle=True)
    i += 1

res = []
for key, val in numpy_vars.items():
    temp = val['arr_0'].tolist()
    res.append(temp)

final_res = {}
for wind in range(111):
    final_res[wind] = []

for wind in range(111):
    for d in res:
        perm = d[0]
        final_res[wind].extend(perm[wind])


final_res_arr = np.array(final_res)
# Now save the 'final_res' object to disk.
np.savez_compressed('G:\jw_lab\jwlab_eeg\Results\\permuted npz collated\\12m_w2v_res_from_egg_null_dist_100x50iters.npz', final_res_arr)

#-------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------

"""
The following section plots two graphs in a single plot. Make sure to import the python packages in the beginning of the file.
"""

def average_fold_accs(data):
    result = {}
    for key, val in data.items():
        result[key] = np.mean(val)

    return result


def significance(h1, h0):
    """"
    This function calculates the p-value of the alternate hypothesis against the
    null distribution.
    """
    """h1 is the alternate hypothesis. This contains the observed values for
    each window which will then be averaged.
    h0 is the null hypothesis. h0 will have many values for one single window.
    The function performs the statistical test with the t-test with the bejamini hochberg correction.
    """

    # The following lines are for debugging purposes only.
    h1 = np.load("G:\jw_lab\jwlab_eeg\\regression\\non_permuted_results\9m_pre_w2v_from_eeg_non_perm.npz",
                 allow_pickle=True)['arr_0'].tolist()
    h0 = np.load("G:\jw_lab\jwlab_eeg\\regression\permuted_npz_processed\kde\9m_pre_w2v_kde_null_dist.npz",
                 allow_pickle=True)['arr_0'].tolist()


    # First we process 'h1' to average all the cross-validation folds accuracies for each window.
    null_h = h0
    alt_h = h1[0]

    alt_h_avg = average_fold_accs(alt_h)

    # Now we perform the significance testing between the 'h1_avg' and 'h0'.
    """
    The p-value is calculated by finding the number of times the permuted accuracies
    are above the observed value (true value).
    The process is done for each window.
    """

    all_perm_accs = []
    for perm_accs in h0.values():
        all_perm_accs.extend(perm_accs)

    denom = len(null_h[0])
    # denom = len(all_perm_accs)
    count_dict = {}
    p_values_list = []  # Stores the window based p-values against the null distribution.
    for window in range(len(alt_h_avg)):
        obs_score = alt_h_avg[window]
        permute_scores = null_h[window]
        # permute_scores = all_perm_accs
        count = 0
        # Now count how many of the permute scores are >= obs_score.
        for j in range(len(permute_scores)):
            if permute_scores[j] > obs_score:
                count += 1
        count_dict[window] = count
        p_value = count / denom
        p_values_list.append(p_value)

    # Implementing the Benjamini-Hochberg correction.
    # First have an index_array just in case.
    idxs = [i for i in range(len(alt_h_avg))]
    # Sort the p_values and idxs in ascending order.
    p_vals_list_asc, p_vals_idx_sort = (list(t) for t in zip(*sorted(zip(p_values_list, idxs))))
    p_vals_asc_rank = [i for i in range(len(alt_h_avg))]

    reject, pvals_corrected, alph_sidak, alph_bonf = multipletests(p_vals_list_asc, is_sorted=True, method='fdr_by', alpha=0.01)

    # The following line is for debugging.
    reject1, pvals_corrected1, alph_sidak, alph_bonf = multipletests(p_values_list, is_sorted=False, method='fdr_by',alpha=0.01)
    p_values_uncorrected = np.array(p_values_list)

    plt.clf()
    sns.set_style("whitegrid")
    sns.lineplot(np.arange(len(p_values_uncorrected)), p_values_uncorrected, label="Uncorrected")
    sns.lineplot(np.arange(len(pvals_corrected1)), pvals_corrected1, label="Corrected", alpha=0.7)
    plt.title("9m pre-w2v from EEG Uncorrected vs Corrected p-values")
    plt.xlabel("Window")
    plt.ylabel("p-value")
    plt.legend()
    plt.show()


    p_vals_idx_sort = np.array(p_vals_idx_sort)

    return reject, pvals_corrected, p_vals_idx_sort



def get_permuted_val(results, h0, ttest=False):
    # h0 = np.load('G:\jw_lab\jwlab_eeg\Results\9m_prew2v_from_egg_null_dist_100x50iters.npz', allow_pickle=True)
    # h0 = h0['arr_0']
    # h0 = h0.tolist()

    null_h = h0
    alt_h = results

    alt_h_avg = average_fold_accs(alt_h)

    if ttest == True:
        null_h_avg = average_fold_accs(null_h)

        return null_h_avg

    denom = len(null_h[0])
    p_values_list = []  # Stores the window based p-values against the null distribution.
    for window in range(len(alt_h_avg)):
        obs_score = alt_h_avg[window]
        permute_scores = null_h[window]
        count = 0
        # Now count how many of the permute scores are >= obs_score.
        for j in range(len(permute_scores)):
            if permute_scores[j] > obs_score:
                count += 1

        p_value = count / denom
        p_values_list.append(p_value)

    return p_values_list

# Display accuracy bokeh_plots graphs side by side.

scoreMean1 = []
scoreMean2 = []
stdev1 = []
stdev2 = []

# In the console: load
#) Assign 9m results to h11, 12m data to h21, 9m null_distribution to h10, and 12m null_distribution values to h20.

h11 = results1
h21 = results2
for i in range(len(results1)):
    for j in range(len(results1[i])):
        scoreMean1.append(round(np.mean(results1[i][j]), 4))
        stdev1.append(round(stats.sem(results1[i][j]), 4))
for i in range(len(results2)):
    for j in range(len(results2[i])):
        scoreMean2.append(round(np.mean(results2[i][j]), 4))
        stdev2.append(round(stats.sem(results2[i][j]), 4))


length_per_window_plt = 10  # 1200 / len(scoreMean)
x_graph = np.arange(-200, 910, length_per_window_plt)
x_graph += 50  # NOTE: Change this based on the window size.
y_graph1 = scoreMean1
y_graph2 = scoreMean2

stdevplt1 = np.array(stdev1)
error1 = stdevplt1
stdevplt2 = np.array(stdev2)
error2 = stdevplt2

# Done: Find the above chance accuracy x1 and x2 values for shading. Do you really need the tvalues_pos?
# You need tvalues_pos if you are shading the region with max t-value. This might be a good idea.
# It also might be a good idea to save the results somewhere to avoid rerunning the experiments everytime.
# max_pos_tvalue_idx = t_mass_pos.index(max(t_mass_pos)) - 1
# # # print("max_pos_tvalue_idx: ", max_pos_tvalue_idx)
# # # print("clusters_pos: ", clusters_pos)
# # print("adj_clusters_pos: ", adj_clusters_pos)

# Running the following line will fail if h1 and h0 are not initialized.
# h0 needs to be initialized for many permutation iterations.
reject1, pvals_corrected1, p_vals_idx_sort1 = significance(h11, h10)
dot_idxs1 = p_vals_idx_sort1[reject1]
x_dots1 = x_graph[dot_idxs1]

reject2, pvals_corrected2, p_vals_idx_sort2 = significance(h21, h20)
dot_idxs2 = p_vals_idx_sort2[reject2]
x_dots2 = x_graph[dot_idxs2]

# Find clusters-1.
pvalues_pos1 = get_permuted_val(results=results1[0], h0=h10, ttest=False)
valid_window_pos1 = [i for i, v in enumerate(pvalues_pos1) if v < 0.05]
clusters_pos1 = [list(group) for group in mit.consecutive_groups(valid_window_pos1)]
clusters_pos1 = [group for group in clusters_pos1 if len(group) >= 3]
adj_clusters_pos1 = []
for c in clusters_pos1:
    new_list = [((x * 10) - 200) for x in c]
    adj_clusters_pos1.append(new_list)

# Find clusters-2
pvalues_pos2 = get_permuted_val(results=results2[0], h0=h20, ttest=False)
valid_window_pos2 = [i for i, v in enumerate(pvalues_pos2) if v < 0.05]
clusters_pos2 = [list(group) for group in mit.consecutive_groups(valid_window_pos2)]
clusters_pos2 = [group for group in clusters_pos2 if len(group) >= 3]
adj_clusters_pos2 = []
for c in clusters_pos2:
    new_list = [((x * 10) - 200) for x in c]
    adj_clusters_pos2.append(new_list)





x11 = 240 + 50#adj_clusters_pos[max_pos_tvalue_idx][0] + 50
x12 = 840 + 50#adj_clusters_pos[max_pos_tvalue_idx][-1] + 50
x21 = -120 + 50
x22 = 580 + 50

# Make sure to change this index to which you wanna retrieve.
# pink_clusters = [list(i) for i in mit.consecutive_groups(sorted(x_graph[p_vals_idx_sort[:27]] / 10))]

y_dots1 = [0.32] * len(x_dots1)
y_dots2 = [0.35] * len(x_dots2)


sns.set() # Setting the style to be default seaborn.
plt.clf()

# plt.rcParams["figure.figsize"] = (20, 15)

plt.plot(x_graph, y_graph1, 'k-', label='9m', color='blue')
plt.plot(x_graph, y_graph2, 'k-', label='12m', color='red')


ylim = [0.3, 0.8]
plt.ylim(ylim[0], ylim[1])
# for cluster in clusters[1:]:
#     x1 = cluster[0] * 10
#     x2 = cluster[-1] * 10
#     plt.fill_betweenx(y=ylim, x1=x1, x2=x2, color="#f5e6fc")

plt.fill_betweenx(y=ylim, x1=x21, x2=x22, color="#fcfa34")
plt.fill_betweenx(y=ylim, x1=x11, x2=x12, color="#f5e6fc")
plt.axhline(0.5, linestyle='--', color='#696969')
plt.axvline(0, linestyle='--', color='#696969')
color1 = '#58c0fc'
color2 = '#784fca'
plt.fill_between(x_graph, y_graph1 - error1, y_graph1 + error1, color=color1)
plt.fill_between(x_graph, y_graph2 - error2, y_graph2 + error2, color=color2)
plt.scatter(x_dots1, y_dots1, marker='.', color=color1)
plt.scatter(x_dots2, y_dots2, marker='.', color=color2)
plt.title("9m 100-10 w2v_res_from_eeg perm 50iter shift-r 50ms")
plt.xlabel("Time (ms)")
plt.ylabel("2v2 Accuracy")
plt.xticks(np.arange(-200, 1001, 200), ['-200', '0', '200', '400', '600', '800', '1000'])
plt.legend(loc=1)
acc_at_zero1 = y_graph1[np.where(x_graph == 0)[0][0]]
acc_at_zero2 = y_graph2[np.where(x_graph == 0)[0][0]]
plt.text(700, 0.35, str(f"9m-Acc At 0ms: {acc_at_zero1}"))
plt.text(700, 0.32, str(f"12m-Acc At 0ms: {acc_at_zero2}"))
plt.show()

#-------------------------------------------------------------------------------------