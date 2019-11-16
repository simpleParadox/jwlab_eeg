import pandas as pd
import numpy as np
from first_participants_map import map_first_participants
from constants import word_list

def prep_ml(filepath, participants):
    df, ys = load_ml_data(filepath, participants)
    return prep_ml_internal(df, ys, participants)

def load_ml_data(filepath, participants):
    # read all participant csvs, concat them into one dataframe
    dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (filepath, s)) for s in participants]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    ys = [np.loadtxt("%s%s_labels.txt" % (filepath, s)) for s in participants]
    return df, ys

def prep_ml_internal(df, ys, participants):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    df = df[df.Time >= 0]
    # we don't want the time column, or the reference electrode, so drop those columns
    df = df.drop(columns=["Time", "E65"], axis=1)

    # now we need to flatten each
    # "block" of data (ie. 1000 rows of 64 columns of eeg data) into one training example, one row
    # of 64*1000 columns of eeg data
    X = df.values
    (i,j) = X.shape
    X = np.reshape(X, (i // 1000, j * 1000))
    
    # map first participants (cel from 1-4 map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ys = map_first_participants(ys, participants)
    y = np.concatenate(ys)

    assert y.shape[0] == X.shape[0]

    # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
    df = pd.DataFrame(data=X)
    df['label'] = y
    df['participant'] = np.concatenate([[ys.index(y)]*len(y) for y in ys])
    
    # remove bad samples
    df = df[df.label != -1]

    # make label zero indexed 
    df.label -= 1

    X,y = no_average(df)

    y[y < 8] = 0
    y[y >= 8] = 1
    
    return X, y

def no_average(df):
    return df.drop(columns=['label', 'participant'], axis=1), df.label.values.flatten()

def average_trials(df):
    num_participants = df.participant.max() + 1
    num_words = len(word_list)

    new_data = np.zeros((num_participants * num_words, 64 * 1000))
    df_data = df.drop(columns=['label', 'participant'], axis=1)
    new_y = np.zeros(num_participants * num_words)
    for p in range(num_participants):
        for w in range(num_words):
            new_data[p * num_words + w, :] = df_data[np.logical_and(df.participant == p, df.label == w)].values.mean()
            new_y[p * num_words + w] = w
    
    return new_data, new_y

def average_trials_and_participants(df):
    num_words = len(word_list)

    new_data = np.zeros((num_words, 64 * 1000))
    df_data = df.drop(columns=['label', 'participant'], axis=1)
    new_y = np.zeros(num_words)
    for w in range(num_words):
        new_data[w, :] = df_data[df.label == w].values.mean()
        new_y[w] = w
    
    return new_data, new_y