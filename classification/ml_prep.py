import pandas as pd
import numpy as np
from first_participants_map import map_first_participants
from constants import word_list
from bad_trials import get_bad_trials, transform_ybad_indices
from scipy.signal import resample

def prep_ml(filepath, participants, downsample_num=1000, averaging="average_trials"):
    df, ys = load_ml_data(filepath, participants)
    return prep_ml_internal(df, ys, participants, downsample_num=downsample_num, averaging=averaging)
# 1000 downsample is average (1000 is our sampling rate so this is no downsample) 

def load_ml_data(filepath, participants):
    # read all participant csvs, concat them into one dataframe
    dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (filepath, s)) for s in participants]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    ys = [np.loadtxt("%s%s_labels.txt" % (filepath, s)) for s in participants]
    return df, ys

def prep_ml_internal(df, ys, participants, downsample_num=1000, averaging="average_trials"):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    df = df[df.Time >= 0]
    # we don't want the time column, or the reference electrode, so drop those columns
    df = df.drop(columns=["Time", "E65"], axis=1)

    # now we need to flatten each
    # "block" of data (ie. 1000 rows of 64 columns of eeg data) into one training example, one row
    # of 64*1000 columns of eeg data
    X = df.values
    X = np.reshape(X, (1000, 64, -1))
    X = resample(X, downsample_num, axis=0)
    (i,j,k) = X.shape
    X = np.reshape(X, (k, j * downsample_num))
        
    # map first participants (cel from 1-4 map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ybad = get_bad_trials(participants, ys)
    ys = map_first_participants(ys, participants)
    y = np.concatenate(ys)
    ybad = transform_ybad_indices(ybad, ys)
    y[ybad] = -1

    assert y.shape[0] == X.shape[0]
    
    # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
    df = pd.DataFrame(data=X)
    df['label'] = y
    df['participant'] = np.concatenate([[ys.index(y)]*len(y) for y in ys])
    
    # remove bad samples
    df = df[df.label != -1]

    # make label zero indexed 
    df.label -= 1

    if averaging == "no_averaging":
        X,y,p,w = no_average(df)
    elif averaging == "average_trials":
        X,y,p,w = average_trials(df)
    else:
        X,y,p,w = average_trials_and_participants(df)

    y[y < 8] = 0
    y[y >= 8] = 1
    
    return X, y, p, w

def no_average(df):
    return df.drop(columns=['label', 'participant'], axis=1), df.label.values.flatten(), df.participant.values, df.label.values

def average_trials(df):
    num_participants = df.participant.max() + 1
    num_words = len(word_list)

    new_data = np.zeros((num_participants * num_words, len(df.columns) - 2))
    df_data = df.drop(columns=['label', 'participant'], axis=1)
    new_y = np.zeros(num_participants * num_words)
    participants = np.zeros(num_participants * num_words)

    for p in range(num_participants):
        for w in range(num_words):
            means = df_data[np.logical_and(df.participant == p, df.label == w)].values.mean()
            new_data[p * num_words + w, :] = means
            new_y[p * num_words + w] = -1 if np.isnan(means).any() else w
            participants[p * num_words + w] = p

    new_data = new_data[new_y != -1, :]
    participants = participants[new_y != -1]
    new_y = new_y[new_y != -1]
    return new_data, new_y, participants, np.copy(new_y)

def average_trials_and_participants(df):
    num_words = len(word_list)

    new_data = np.zeros((num_words, len(df.columns) - 2))
    df_data = df.drop(columns=['label', 'participant'], axis=1)
    new_y = np.zeros(num_words)
    for w in range(num_words):
        new_data[w, :] = df_data[df.label == w].values.mean()
        new_y[w] = w
    
    return new_data, new_y, np.ones(num_words) * -1, np.copy(new_y)