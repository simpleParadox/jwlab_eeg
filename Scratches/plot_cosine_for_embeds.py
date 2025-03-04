import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import os
import seaborn as sns
import pandas as pd
import glob



# First 8 are animate and the last 8 are inanimate.
labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}


def plot_cosine_distances(cosine_distances, title, labels=None):
    fig, ax = plt.subplots()
    im = ax.imshow(cosine_distances, cmap='seismic', vmin=0, vmax=1)
    # im = ax.imshow(cosine_distances, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    # cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=10, size=1)
    cbar.set_label('Cosine Distance x 10^3', rotation=270, labelpad=15, fontsize=10)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0,
             rotation_mode="anchor")
    # for i in range(len(labels_mapping)):
    #     for j in range(len(labels_mapping)):
    #         text = ax.text(j, i, cosine_distances[i, j],
    #                        ha="center", va="center", color="w")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    plt.show()

def calculate_cosine_distance(a, b):
    return cosine(a, b)

def get_cosine_distance_for_all_stims(stims):
    cosine_distances = np.zeros((16, 16))
    for i in range(len(stims)):
        for j in range(len(stims)):
            cosine_distances[i,j] = calculate_cosine_distance(stims[i], stims[j])
    return cosine_distances


def load_embeds(embeds_path):
    embeds = np.load(embeds_path, allow_pickle=True)['arr_0'].tolist()
    return embeds


def get_cosine_from_excel(excel_path):
    df = pd.read_excel(excel_path, index_col=0)
    return df


# embeds_with_label_path = os.getcwd() + "/regression/w2v_embeds/embeds_with_label_dict.npz"
age = 12
all_iter_embeds = []
for embeds_with_label_path in glob.glob(f"/Users/simpleparadox/PycharmProjects/jwlab_eeg/regression/predictions/{age}m*.npz"):
    embeds = np.load(embeds_with_label_path, allow_pickle=True)['arr_0'][0]
    all_iter_embeds.append(embeds)

all_iter_embeds = np.array(all_iter_embeds)
mean_all_iter_embeds = np.mean(all_iter_embeds, axis=0)
embeds = mean_all_iter_embeds


# embeds = list(load_embeds(embeds_with_label_path)[0].values())

cosine_distances = get_cosine_distance_for_all_stims(embeds)

df = pd.DataFrame(cosine_distances, columns=labels_mapping.values(), index=labels_mapping.values())
filepath = os.getcwd() + f"/regression/predictions/{age}m_cosine_distances_preds_pre_w2v_with_label_dict_mat.xlsx"
df.to_excel(filepath)

# plot_cosine_distances(cosine_distances, "Cosine distance between all predicted embeddings")

cosine_distances_from_excel = get_cosine_from_excel(filepath)

# Change the scale of the values in the predicted cosine distance matrix. Multiply by 1000.
cosine_distances_from_excel = cosine_distances_from_excel * 10**3

plot_cosine_distances(cosine_distances_from_excel, "Cosine-dist b/w predicted pre-w2v 12m scaled 10^3 ", labels=cosine_distances_from_excel.columns.values)


plt.clf()
fig = plt.figure(figsize=(12, 10))
sns.heatmap(cosine_distances_from_excel)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=10)
plt.title("Cosine distance between all stims ", fontsize=25)
plt.tick_params(axis='both', labelsize=18)
plt.show()