import numpy as np
temp = np.load("G:\jw_lab\jwlab_eeg\\regression\cosines\cosine_sim_9m_results_with_mean_ytrue_vec.npz", allow_pickle=True)
score_iters = temp['arr_0'].tolist()

window_scores = {}
for run, scores in enumerate(score_iters):
    window_scores[run] = scores[0]

appended_window_scores = {}
for i in range(len(window_scores[0])):
    appended_window_scores[i] = []

for window in range(len(window_scores[0])):
    for run, scores in window_scores.items():
        window_all_word_scores = scores[window]  # Get the scores for that window.
        appended_window_scores[window].append([window_all_word_scores])

global_flat_wind_scores = []
for key, wind_score in appended_window_scores.items():
    a = wind_score
    flattened_wind_score = []
    for j in range(len(a)):
        flattened_wind_score.append(a[0][0])
    global_flat_wind_scores.append(np.mean(flattened_wind_score, axis=0))

labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}

final_word_scores = {}
for i in range(len(labels_mapping)):
    final_word_scores[labels_mapping[i]] = []
for i in range(len(labels_mapping)):
    for j in range(len(global_flat_wind_scores)):
        final_word_scores[labels_mapping[i]].append(global_flat_wind_scores[j][i])


np.savez_compressed("G:\jw_lab\jwlab_eeg\\regression\cosines\\processed_cosine_scores_dict_9m_pre_and_post_onset.npz", final_word_scores)
    


# Plotting the cosine distance for the pre-onset window.
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np

# pre_onset_data = np.load("G:\jw_lab\jwlab_eeg\\regression\cosines\processed_cosine_scores_dict_9m_pre_onset.npz", allow_pickle=True)['arr_0'].tolist()
# post_onset_data = np.load("G:\jw_lab\jwlab_eeg\\regression\cosines\processed_cosine_scores_dict_9m_post_onset.npz", allow_pickle=True)['arr_0'].tolist()
data = np.load("G:\jw_lab\jwlab_eeg\\regression\cosines\cosine_sim_9m_results_with_mean_ytrue_vec.npz", allow_pickle=True)['arr_0'].tolist()
x_graph = np.arange(-200, 910, 10)
x_graph += 100
plt.clf()
NUM_COLORS = 16
cm = plt.get_cmap('gist_rainbow')
cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
i = 0
plt.axvline(x=0, color='grey', linestyle='dashed')
for word, cosine_scores in data.items():
    y_data = []
    y_data.extend(data[word])
    lines = ax.plot(x_graph, y_data, label=word)
    if i > 7:
        lines[0].set_linestyle('dashed')
    i += 1

plt.legend(bbox_to_anchor=(1, 1.05))
plt.title("Cosine Similarities 9m predicting w2v from EEG")
plt.xticks(np.arange(-200, 1001, 200), ['-200', '0', '200', '400', '600', '800', '1000'])
plt.xlabel("Time (ms)")
plt.ylabel("Cosine Similarity")
plt.show()