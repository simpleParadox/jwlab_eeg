import pandas as pd
import numpy as np
from first_participants_map import map_first_participants
from constants import word_list

def prep_ml(filepath, participants):
    # read all participant csvs, concat them into one dataframe
    dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (filepath, s)) for s in participants]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    df = df[df.Time >= 0]
    # we don't want the time column, or the reference electrode, so drop those columns
    df = df.drop(columns=["Time", "E65"], axis=1)

    # finally for X, we need to turn the dataframe into a numpy array, and then flatten each
    # "block" of data (ie. 1000 rows of 64 columns of eeg data) into one training example, one row
    # of 64*1000 columns of eeg data
    X = df.values
    (i,j) = X.shape
    X = np.reshape(X, (i // 1000, j * 1000))

    # load all labels, do any extra mapping required, and concat them into one array of y's
    ys = [np.loadtxt("%s%s_labels.txt" % (filepath, s)) for s in participants]
    
    ys = map_first_participants(ys, participants)

    assert np.concatenate(ys).shape[0] == X.shape[0]
    
    X, ys = remove_bad_samples(X, ys)
    ys = make_zero_indexed(ys)

    y = np.concatenate(ys)
    X,y = average_trials(y, X)
    
    return X, y

def remove_bad_samples(X, ys):
    Xp = X[np.concatenate(ys) != -1, :]
    
    for i in range(len(ys)):
        ys[i] = np.array(ys[i])
        ys[i] = ys[i][ys[i] != -1]
    return Xp, ys

def make_zero_indexed(ys):
    for i in range(len(ys)):
        ys[i] = ys[i] - 1
    return ys

def average_trials(ys, X):
    (n,d) = X.shape
    Xp = np.zeros((n // len(word_list), d))
    y = np.zeros(n // len(word_list))
    i = 0
    for y in ys:
        print(len(np.unique(y)))
        assert len(np.unique(y)) == len(word_list)

        for v in range(len(word_list)):
            Xp[v, :] = np.mean(X[y == v,:], axis=1)
            y[i] = v
            i = i + 1
    return Xp, y
