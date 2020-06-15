import pandas as pd
import numpy as np
from jwlab.ml_prep import  average_trials_and_participants, no_average, average_trials
from jwlab.data_graph import plot_good_trial_participant, plot_good_trial_word
from jwlab.participants_map import map_participants
from jwlab.constants import word_list, bad_trials_filepath, old_participants
from jwlab.bad_trials import get_bad_trials, get_left_trial_each_word


def slide_df(df, length_per_window):
    num_windows = int(1000 / length_per_window) 
    windows_list = []
    for i in range(num_windows):
        windows_list.append(df[(df.Time < ((i+1) * length_per_window)) & (df.Time >= i * length_per_window)])
    assert len(windows_list) == num_windows
    return windows_list

def load_ml_data(filepath, participants):
    # read all participant csvs, concat them into one dataframe
    dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (filepath, s))
           for s in participants]
    df = pd.concat(dfs, axis=0, ignore_index=True, sort=True)

    ys = [np.loadtxt("%s%s_labels.txt" % (filepath, s)).tolist()
          for s in participants]
    print("loaded", flush=True)
    return df, ys

def prep_cluster_analysis(filepath, participants, downsample_num=1000, averaging="average_trials_and_participants", length_per_window=10):
    df, ys = load_ml_data(filepath, participants)
    return prep_cluster_analysis_internal(df, ys, participants, downsample_num=downsample_num, averaging=averaging, length_per_window=length_per_window)


def prep_cluster_analysis_internal(df, ys, participants, downsample_num=1000, averaging="average_trials_and_participants", length_per_window=10):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    df = df[df.Time >= 0]

    # map first participants (cel from 1-4 map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ybad = get_bad_trials(participants)
    ys = map_participants(ys, participants)

    # set the value of bad trials in ys_curr to -1 (to exclude from learning)
    trial_count = []
    bad_trial_count = []
    for each_ps in range(len(ys)):
        for bad_trial in range(len(ybad[each_ps])):
            # ys_curr[each_ps]: for the total trial sub-list of each participant of ys_curr...
            # ybad[each_ps][bad_trial]: for each trial index in the bad trial sub-list of each participant of ybad...
            # minus 1 since in ys_curr trials are zero-indexed while in bad_trial it's one-indexed (because they are directly read from csv)
            ys[each_ps][ybad[each_ps][bad_trial]-1] = -1

        # count the total number of trials for each participant
        trial_count += [len(ys[each_ps])]
        bad_trial_count += [len(ybad[each_ps])]

    # good trial each participant 
    good_trial_participant_count = np.around(np.true_divide(
        np.subtract(trial_count, bad_trial_count), trial_count), decimals=2)
    # good trial each word each participant
    good_trial_word_count = get_left_trial_each_word(participants)

    Y = np.concatenate(ys)

    windows_list = slide_df(df, length_per_window)

    X_list = [0] * int(1000 / length_per_window)
    y_list = X_list[:]
    p_list = X_list[:]
    w_list = X_list[:]
    df_list = X_list[:]
    
    for each_window in range(len(windows_list)):
        df = windows_list[each_window]
        df = df.drop(columns=["Time", "E65"], axis=1)
        X = df.values
        X = np.reshape(
            X, (length_per_window, 60, -1))
        #X = resample(X, downsample_num, axis=0)
        (i, j, k) = X.shape
        X = np.reshape(X, (k, j * length_per_window))
      
        

        # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
        df = pd.DataFrame(data=X)
        df['label'] = Y
        df['participant'] = np.concatenate(
            [[ys.index(y)]*len(y) for y in ys])

        # remove bad samples
        df = df[df.label != -1]

        # make label zero indexed
        df.label -= 1
        
#         # get the first 20 rows of each participant
#         df = df.groupby('participant').head(20)

        # different averaging processes
        if averaging == "no_averaging":
            X, y, p, w = no_average(df)
        elif averaging == "average_trials":
            X, y, p, w = average_trials(df)
        elif averaging == "average_trials_and_participants":
            X, y, p, w = average_trials_and_participants(df, participants)
        else:
            raise ValueError("Unsupported averaging!")

        y[y < 8] = 0
        y[y >= 8] = 1
        
        X_list[each_window] = X
        y_list[each_window] = y
        p_list[each_window] = p
        w_list[each_window] = w
        df_list[each_window] = df
                
    return X_list, y_list, [good_trial_participant_count, good_trial_word_count]