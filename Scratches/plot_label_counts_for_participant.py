# Import packages for plotting.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}

labels_mapping_percentage_age_group = 12
labels_mapping_percentage_9m = {
    'baby': 6/58,
    'bear': 6/63,
    'bird': 8/68,
    'bunny': 5/54,
    'cat': 7/61,
    'dog': 6/59,
    'duck': 8/63,
    'mom': 6/68,
    'banana': 6/60,
    'bottle': 8/55,
    'cookie': 5/62,
    'cracker': 7/59,
    'cup': 6/70,
    'juice': 6/57,
    'milk': 7/57,
    'spoon': 6/60
}
labels_mapping_percentage_12m = {
    'baby': 6/41,
    'bear': 8/44,
    'bird': 8/42,
    'bunny': 8/46,
    'cat': 6/39,
    'dog': 7/42,
    'duck': 8/40,
    'mom': 8/37,
    'banana': 6/37,
    'bottle': 8/47,
    'cookie': 6/40,
    'cracker': 8/46,
    'cup': 7/39,
    'juice': 7/42,
    'milk': 7/39,
    'spoon': 7/43
}
my_dict = labels_mapping_percentage_12m
if labels_mapping_percentage_age_group == 9:
    my_dict = labels_mapping_percentage_9m


# Plot the percentage of each label in the dataset.


keys = list(my_dict.keys())

# get values in the same order as keys, and parse percentage values
vals = [float(my_dict[k]) for k in keys]
plt.clf()
plt.rcParams.update({'font.size': 12})
sns.barplot(x=keys, y=vals)

# Store the values for the labels in a list.
labels = list(labels_mapping.values())
plt.xticks(rotation=90)
plt.xlabel("Word")
plt.ylabel(f"Max % coming from one participant - {labels_mapping_percentage_age_group}m old")
# plt.title(f"Maximum % contribution for each word - {labels_mapping_percentage_age_group}-month-old")
plt.tight_layout()
# plt.show()
plt.savefig(f"percent_contribution_max_for_each_word_{labels_mapping_percentage_age_group}m.png")



# Read the data.
age_group = 12
data = pd.read_csv(f"/Users/simpleparadox/PycharmProjects/jwlab_eeg/regression/{age_group}m_df_ch_group_0.csv")
data = data[['participant', 'label']]

group_label_value_counts = data['label'].value_counts()
# Plot the value counts with a bar chart.
plt.clf()

plt.figure(figsize=(12, 8))
# Set the font size to 12.
plt.rcParams.update({'font.size': 12})
sns.barplot(x=group_label_value_counts.index, y=group_label_value_counts.values, palette="tab20")
# plt.title(f"Total number of trials for each word - {age_group}-month-olds", fontsize=20)
plt.ylabel(f"Total trial count - {age_group}m old", fontsize=12)
plt.xlabel("Word", fontsize=12)
# change the axis to show the word labels.
plt.xticks(range(len(labels)), labels, rotation=45)
#
# plt.show()
plt.savefig(f"pooled_label_counts_for_all_{age_group}m.png")


temp = data.groupby(['participant', 'label']).size().reset_index(name='counts')


# Create a new dataframe with the index as the participant and the columns as the labels counts.
new_df = pd.DataFrame(index=temp['participant'].unique(), columns=labels)

# Fill the new dataframe with the counts.
for index, row in temp.iterrows():
    new_df.loc[row['participant'], labels_mapping[int(row['label'])]] = row['counts']



# For each row in the dataframe, find the label with the highest count.



# fill the NaNs with 0.
numeric_df = new_df.fillna(0)

# Convert the columns to numeric.
numeric_df = numeric_df.apply(pd.to_numeric)

# Get the indices of the value that is the max for each row.
numeric_df['max_label'] = numeric_df.idxmax(axis=1)

max_counts_per_participant = {}
for index, row in numeric_df.iterrows():
    max_counts_per_participant[index + 1] = [row['max_label'], row[row['max_label']]]


# Create a dataframe with the max counts per participant along with the label.
max_counts_per_participant_df = pd.DataFrame.from_dict(max_counts_per_participant, orient='index', columns=['word', 'count'])
max_counts_per_participant_df['participant'] = max_counts_per_participant_df.index
# Move the participant column to the front.
cols = max_counts_per_participant_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
max_counts_per_participant_df = max_counts_per_participant_df[cols]



# plt.clf()
# bar = max_counts_per_participant_df.plot(x='participant', y='count', kind='bar')
# plt.bar_label(bar.containers[0], labels=max_counts_per_participant_df['word'], label_type='edge', rotation=45)
# plt.ylabel('Count')
# plt.xlabel('Participant')
# plt.title(f'Max Label Counts for {age_group}m')
# plt.show()

# Increase the index by 1.
new_df.index += 1
plt.clf()
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(15, 8))
# sns.set()

# Create a cmap with 30 unique colors.
cmap = sns.color_palette("hls", 30)
# Plot the bar chart.
plt.rcParams.update({'font.size': 12})

# N = 10 # number of colors to extract from each of the base_cmaps below
# base_cmaps = ['Greys','Purples','Reds','Blues','Oranges','Greens']
# n_base = len(base_cmaps)
# # we go from 0.2 to 0.8 below to avoid having several whites and blacks in the resulting cmaps
# colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.2,0.8,N)) for name in base_cmaps])
# cmap = matplotlib.colors.ListedColormap(colors)
colors = ["#000000", "#696969", "#d3d3d3", "#191970", "#8b0000", "#808000", "#3cb371", "#ff0000", "#ff8c00",
          "#ffd700", "#0000cd", "#00ff7f", "#00ffff", "#00bfff", "#adff2f", "#ff00ff", "#f0e68c", "#fa8072", "#dda0dd",
          "#ff1493", "#7b68ee"]
new_df.T.plot(kind='bar', stacked=True, ax=ax, color=colors[:new_df.shape[0]])
plt.legend(new_df.index, loc='center left', bbox_to_anchor=(1.0, 0.5), title="Participant")
# plt.title(f"Label Counts for Each Participant - {age_group} Month group", fontsize=26)
plt.ylabel("Count", fontsize=12)
plt.xlabel("Word", fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig(f"word_dist_for_{age_group}m.png")


# # Plot the counts for the labels for each participant using a seaborn stacked bard chart.
# plt.clf()
# fig, ax = plt.subplots(figsize=(15, 8))
# sns.barplot(data=temp, x='participant', y='counts', hue='label', ax=ax, stack=True)
# ax.set_title(f"Label Counts for Age Group {age_group} Months")
# ax.set_xlabel("Participant")
# ax.set_ylabel("Count")
# plt.show()