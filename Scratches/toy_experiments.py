import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeavePOut


from regression.functions import extended_2v2_mod

age_group = 9
data = pd.read_csv(f"/Users/simpleparadox/PycharmProjects/jwlab_eeg/regression/{age_group}m_df_ch_group_0.csv")
data = data[['participant', 'label']]

group_label_value_counts = data['label'].value_counts()

# Read in the word vectors.
label_count = []
word_vectors = []
word_vectors_data = np.load("/Users/simpleparadox/PycharmProjects/jwlab_eeg/regression/w2v_embeds/embeds_with_label_dict.npz", allow_pickle=True)['arr_0'][0]
for label, word_vector in word_vectors_data.items():
    word_vectors.append(word_vector)
    label_count.append(group_label_value_counts[label])
word_vectors = np.array(word_vectors)
label_count = np.array(label_count).reshape(-1, 1)

# Print out the shapes.
print(f"word_vectors shape: {word_vectors.shape}")
print(f"label_count shape: {label_count.shape}")


# Do leave two out cross validation.
lpo = LeavePOut(p=2)

all_test_scores = []
for i, (train_index, test_index) in enumerate(lpo.split(word_vectors, label_count)):
    X_train = label_count[train_index]
    y_train = word_vectors[train_index]
    X_test = label_count[test_index]
    y_test = word_vectors[test_index]

    # Train a linear regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)


    # Calculate the 2 vs 2 accuracy.
    _, _, testScore, _, _, _, _, _, _, _ = extended_2v2_mod(y_test, preds)
    all_test_scores.append(testScore)

print(f"Mean 2v2 accuracy: {np.mean(all_test_scores)}")









# Experiment two with the inputs as vector of participant counts and the output as the word vector.


labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}
labels = list(labels_mapping.values())
age_group = 9
data = pd.read_csv(f"/Users/simpleparadox/PycharmProjects/jwlab_eeg/regression/{age_group}m_df_ch_group_0.csv")
data = data[['participant', 'label']]
temp = data.groupby(['participant', 'label']).size().reset_index(name='counts')

new_df = pd.DataFrame(index=temp['participant'].unique(), columns=labels)

# Fill the new dataframe with the counts.
for index, row in temp.iterrows():
    new_df.loc[row['participant'], labels_mapping[int(row['label'])]] = row['counts']
new_df = new_df.fillna(0)

# Add a row to new_df that is the sum of the specific column.
### new_df.loc['sum'] = new_df.sum(axis=0)

# Define the inputs and outputs.

label_count = []
word_vectors = []
word_vectors_data = np.load("/Users/simpleparadox/PycharmProjects/jwlab_eeg/regression/w2v_embeds/embeds_with_label_dict.npz", allow_pickle=True)['arr_0'][0]
for column_num in range(len(new_df.columns)):
    word_vectors.append(word_vectors_data[column_num])
word_vectors = np.array(word_vectors)
X = new_df.T.values


# Do leave two out cross validation.
lpo = LeavePOut(p=2)

# all_test_scores = []
# for i, (train_index, test_index) in enumerate(lpo.split(X, word_vectors)):
X_train = X#[train_index]
y_train = word_vectors#[train_index]
X_test = X#[test_index]
y_test = word_vectors#[test_index]

# Train a linear regression model.
model = Ridge()
model.fit(X_train, y_train)
preds = model.predict(X_test)


# Calculate the 2 vs 2 accuracy.
_, _, testScore, _, _, _, _, _, _, _ = extended_2v2_mod(y_test, preds)
# all_test_scores.append(testScore)

print("2v2 accuracy: ", testScore)
# print(f"Mean 2v2 accuracy: {np.mean(all_test_scores)}")


