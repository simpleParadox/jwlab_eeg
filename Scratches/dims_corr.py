"""
This script calculates the correlation between the columns of the predicted embeddings and the true embeddings.
"""

import numpy as np
import scipy.stats as st
window_preds = np.load("G:\jw_lab\jwlab_eeg\\regression\pred_embeds\9m_w2v_preds.npz", allow_pickle=True)["arr_0"].tolist()[0]
true_vecs_dict = np.load("G:\jw_lab\jwlab_eeg\\regression\w2v_embeds\embeds_with_label_dict.npz", allow_pickle=True)['arr_0'].tolist()[0]

# First store the embeddings into an array.
true_vecs = np.array([v for v in true_vecs_dict.values()])


# First average the preds for all the runs for each window.
avg_window_pred = []
for i in range(len(window_preds)):
    window_runs = window_preds[i]
    avg_window_pred.append(np.mean(window_runs, axis=0))

# %time
column_corrs = []
for window in range(len(avg_window_pred)):
    # print("Window: {0}".format(window))
    pred_vector = avg_window_pred[window]
    window_corr = []
    for j in range(300):
        corr = st.pearsonr(pred_vector[:, j], true_vecs[:, j])
        window_corr.append(corr)
    column_corrs.append(window_corr)


# Now make a violin plot. Use Seaborn?
import seaborn as sns
import matplotlib.pyplot as plt
sns.violinplot(data=column_corrs)
plt.show()