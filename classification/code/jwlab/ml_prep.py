import pandas as pd
import numpy as np
from jwlab.first_participants_map import map_first_participants
from jwlab.constants import word_list, bad_trials_filepath
from jwlab.bad_trials import get_bad_trials, transform_ybad_indices
from scipy.signal import resample

def create_ml_df(filepath, participants, downsample_num=1000):
    df, ys = load_ml_data(filepath, participants)
    return create_ml_df_internal_sktime(df, ys, participants, downsample_num=downsample_num)

def load_ml_data(filepath, participants):
    # read all participant csvs, concat them into one dataframe
    dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (filepath, s)) for s in participants]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    ys = [np.loadtxt("%s%s_labels.txt" % (filepath, s)) for s in participants]
    print("loaded", flush=True)
    return df, ys

def prep_ml(filepath, participants, downsample_num=1000, averaging="average_trials"):
    df, ys = load_ml_data(filepath, participants)
    return prep_ml_internal(df, ys, participants, downsample_num=downsample_num, averaging=averaging)

def prep_ml_internal(df, ys, participants, downsample_num=1000, averaging="average_trials"):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    df = df[df.Time >= 0]
    # we don't want the time column, or the reference electrode, so drop those columns
    df = df.drop(columns=["Time", "E65", "E64", "E63", "E62", "E61"], axis=1)
    # now we need to flatten each
    # "block" of data (ie. 1000 rows of 64 columns of eeg data) into one training example, one row
    # of 64*1000 columns of eeg data
    X = df.values
    X = np.reshape(X, (1000, 60, -1))
    X = resample(X, downsample_num, axis=0)
    (i,j,k) = X.shape
    X = np.reshape(X, (k, j * downsample_num))


    # map first participants (cel from 1-4 map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ybad, bad_trial_count = get_bad_trials(participants, ys, bad_trials_filepath)
    ys = map_first_participants(ys, participants)
    trial_count = []
    for each_ps in range(len(ys)):
        for bad_trial in range(len(ybad[each_ps])):
            ys[each_ps][ybad[each_ps][bad_trial]-1] = -1
        trial_count += [len(ys[each_ps])]
    
    for i in range(len(participants)):
        print("The number of good trials left for participant - [%s] is - [%d]." % (participants[i], trial_count[i] - bad_trial_count[i]))
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

    if averaging == "no_averaging":
        X,y,p,w = no_average(df)
    elif averaging == "average_trials":
        X,y,p,w = average_trials(df)
    else:
        X,y,p,w = average_trials_and_participants(df)

    y[y < 8] = 0
    y[y >= 8] = 1
    
    return X, y, p, w, df

def create_ml_df_internal(df, ys, participants, downsample_num=1000, bad_trials_filepath=bad_trials_filepath):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    df = df[df.Time >= 0]
    # we don't want the time column, or the reference electrode, so drop those columns
    df = df.drop(columns=["Time", "E65", "E64", "E63", "E62", "E61"], axis=1)

    # now we need to flatten each
    # "block" of data (ie. 1000 rows of 64 columns of eeg data) into one training example, one row
    # of 64*1000 columns of eeg data
    X = df.values
    X = np.reshape(X, (1000, 60, -1))
    X = resample(X, downsample_num, axis=0)
    (i,j,k) = X.shape
    X = np.reshape(X, (k, j * downsample_num))
        
    # map first participants (cel from 1-4 map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ybad = get_bad_trials(participants, ys, bad_trials_filepath)
    ys = map_first_participants(ys, participants)
    for each_ps in range(len(ys)):
        for bad_trial in range(len(ybad[each_ps])):
            ys[each_ps][ybad[each_ps][bad_trial]-1] = -1
    y = np.concatenate(ys)

    assert y.shape[0] == X.shape[0]
    # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
    df = pd.DataFrame(data=X)
    df['label'] = y
    df['participant'] = np.concatenate([[i]*len(y) for i,y in enumerate(ys)])
    
    # remove bad samples
    df = df[df.label != -1]

    # make label zero indexed 
    df.label -= 1

    return df

def create_ml_df_internal_sktime(df, ys, participants, downsample_num=1000, bad_trials_filepath=bad_trials_filepath):    
    df = df[df.Time >= 0]
    df = df.drop(columns=["Time", "E65", "E64", "E63", "E62", "E61"], axis=1)

    df['id'] = np.concatenate([[i] * 1000 for i in range(len(df.index) // 1000)])
    df = df.groupby(["id"], as_index=False).agg(pd.Series)
            
    # map first participants (cel from 1-4 map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ybad = get_bad_trials(participants, ys, bad_trials_filepath)
    ys = map_first_participants(ys, participants)
    for each_ps in range(len(ys)):
        for bad_trial in range(len(ybad[each_ps])):
            ys[each_ps][ybad[each_ps][bad_trial]-1] = -1
    y = np.concatenate(ys)
        
    # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
    #df = pd.DataFrame(data=X)
    df['label'] = y
    df['participant'] = np.concatenate([[ys.index(y)]*len(y) for y in ys])
        
    # remove bad samples
    df = df[df.label != -1]

    # make label zero indexed 
    df.label -= 1
    return df

def save_ml_df(df, filepath):
    df.to_pickle(filepath)

def load_ml_df(filepath):
    return pd.read_pickle(filepath)

def y_to_binary(y):
    y[y < 8] = 0
    y[y >= 8] = 1
    return y

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
            means = df_data[np.logical_and(df.participant == p, df.label == w)].values.mean() if df_data[np.logical_and(df.participant == p, df.label == w)].size != 0 else 0
            new_data[p * num_words + w, :] = means
            new_y[p * num_words + w] = -1 if np.isnan(means).any() else w
            participants[p * num_words + w] = p

    #new_data = new_data[new_y != -1, :]
    #participants = participants[new_y != -1]
    #new_y = new_y[new_y != -1]
    return new_data, new_y, participants, np.copy(new_y)

def average_trials_and_participants(df):
    num_participants = df.participant.max() + 1
    num_words = len(word_list)

    new_data = np.zeros((num_words, len(df.columns) - 2))
    df_data = df.drop(columns=['label', 'participant'], axis=1)
    new_y = np.zeros(num_words)

    for w in range(num_words):
        means = pd.DataFrame([df_data[np.logical_and(df.participant == p, df.label == w)].values.mean() for p in range(num_participants)])
        new_data[w, :] = means.values.mean()
        new_y[w] = -1 if np.isnan(means).any() else w

    new_data = new_data[new_y != -1, :]
    new_y = new_y[new_y != -1]
    
    return new_data, new_y, np.ones(new_y.shape[0]) * -1, np.copy(new_y)