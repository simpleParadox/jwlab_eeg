import math
import random
import pandas as pd
import numpy as np
from scipy.signal import resample
from IPython.display import display
from jwlab.data_graph import plot_good_trial_participant, plot_good_trial_word
from jwlab.participants_map import map_participants
from jwlab.bad_trials import get_bad_trials, get_left_trial_each_word
from jwlab.constants import word_list, bad_trials_filepath, old_participants, cleaned_data_filepath

################################ prep data ################################

def init(age_group):
    if age_group is 9:
#         participants = ["904", "905", "906", "908", "909","910", "912", "913", "914",  "916", "917", "921", "923", "927", "929", "930", "932"] 

#all
        participants = [ "904", "905","906", "908", "909", "912", "913", "914", "916", "917", "919", "920", "921", "923", "924", "927", "929","928", "930", "932"]  
    elif age_group is 11:
        participants = ["106", "107", "109", "111", "112", "115", "116", "117", "119", "120", "121", "122", "124"]
    else:
        raise ValueError("Unsupported age group!")
        
    return participants

def load_ml_data(participants):
    # read all participant csvs, concat them into one dataframe
    dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (cleaned_data_filepath, s))
           for s in participants]
    df = pd.concat(dfs, axis=0, ignore_index=True, sort=True)

    ys = [np.loadtxt("%s%s_labels.txt" % (cleaned_data_filepath, s)).tolist()
          for s in participants]
    print("loaded", flush=True)
    return df, ys

def prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000):
    participants = init(age_group)
    df, ys = load_ml_data(participants)
    return prep_ml_internal(df, ys, participants, useRandomizedLabel, averaging, sliding_window_config, downsample_num=downsample_num)

def prep_ml_internal(df, ys, participants, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    # df = df[df.Time >= 0]
    
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
    
    if useRandomizedLabel:
        random.shuffle(Y)
        print("labels shuffled")
        
    #### Sliding window section ####
    start_time = sliding_window_config[0]
    end_time = sliding_window_config[1]
    window_lengths = sliding_window_config[2]
    step_length = sliding_window_config[3]
        
    windows_list, num_win = slide_df(df, start_time, end_time, window_lengths, step_length)
    
    X_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]
    y_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]   
    p_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]   
    w_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]  
    df_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]
    
    for length_per_window in range(len(windows_list)):
        for each_window in range(len(windows_list[length_per_window])):
            df = windows_list[length_per_window][each_window]
            df = df.drop(columns=["Time", "E65"], axis=1)
            X = df.values
            X = np.reshape(
                X, (window_lengths[length_per_window], 60, -1))
            #X = resample(X, downsample_num, axis=0)
            (i, j, k) = X.shape
            X = np.reshape(X, (k, j * window_lengths[length_per_window]))

            # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
            df = pd.DataFrame(data=X)
            df['label'] = Y
            df['participant'] = np.concatenate(
                [[ys.index(y)]*len(y) for y in ys])

            # remove bad samples
            df = df[df.label != -1]

            # make label zero indexed
            df.label -= 1
            
            # different averaging processes
            if averaging == "no_averaging":
                X, y, p, w = no_average(df)
            elif averaging == "average_trials":
                X, y, p, w = average_trials(df)
            elif averaging == "average_trials_and_participants":
                X, y, p, w = average_trials_and_participants(df, participants)
            elif averaging == "permutation":
                ## 5 is the averaging set size
                df = permutation_and_average(df, 20)
                X, y, p, w = no_average(df)
            else:
                raise ValueError("Unsupported averaging!")
## binary: animacy
            y[y < 8] = 0
            y[y >= 8] = 1

#     #mom and baby vs all
#             y[y == 0] = 1
#             y[y == 1] = 0
#             y[y == 2] = 0
#             y[y == 3] = 0
#             y[y == 4] = 0
#             y[y == 5] = 0
#             y[y == 6] = 0
#             y[y == 7] = 1
            
#             y[y == 8] = 0
#             y[y == 9] = 0
#             y[y == 10] = 0
#             y[y == 11] = 0
#             y[y == 12] = 0
#             y[y == 13] = 0
#             y[y == 14] = 0
#             y[y == 15] = 0

#             #new groups 
#             #0: people
#             #1: animals
#             #2: food
#             #3: kitchen objects
            
#             y[y == 0] = 0
#             y[y == 1] = 1
#             y[y == 2] = 1
#             y[y == 3] = 1
#             y[y == 4] = 1
#             y[y == 5] = 1
#             y[y == 6] = 1
#             y[y == 7] = 0
            
#             y[y == 8] = 2
#             y[y == 9] = 3
#             y[y == 10] = 2
#             y[y == 11] = 2
#             y[y == 12] = 3
#             y[y == 13] = 2
#             y[y == 14] = 2
#             y[y == 15] = 3

            X_list[length_per_window][each_window] = X
            y_list[length_per_window][each_window] = y
            p_list[length_per_window][each_window] = p
            w_list[length_per_window][each_window] = w
            df_list[length_per_window][each_window] = df
        
    return X_list, y_list, [good_trial_participant_count, good_trial_word_count], num_win

# Raw data
def no_average(df):
    return df.drop(columns=['label', 'participant'], axis=1).values, df.label.values.flatten(), df.participant.values, df.label.values

# For each participant, average the value for each word. Expected shape[0] is len(participants) x len(word_list)
def average_trials(df):
    num_participants = df.participant.max() + 1
    num_words = len(word_list)

    new_data = np.zeros((num_participants * num_words, len(df.columns) - 2))
    df_data = df.drop(columns=['label', 'participant'], axis=1)
    new_y = np.zeros(num_participants * num_words)
    participants = np.zeros(num_participants * num_words)

    for p in range(num_participants):
        for w in range(num_words):
            means = df_data[np.logical_and(df.participant == p, df.label == w)].values.mean(axis=0
            ) if df_data[np.logical_and(df.participant == p, df.label == w)].size != 0 else 0
            new_data[p * num_words + w, :] = means
            new_y[p * num_words + w] = -1 if np.isnan(means).any() else w
            participants[p * num_words + w] = p
    return new_data, new_y, participants, np.copy(new_y)

# Average the value of each word across participants. Expected shape[0] is len(word_list)
def average_trials_and_participants(df, participants):
    num_words = len(word_list)
    data, y, participants_rt, w = average_trials(df)
    new_data = np.zeros((num_words, len(df.columns) - 2))
    new_y = np.zeros(num_words)
    for w in range(num_words):
        count = 0
        for p in range(len(participants)):
            count += data[p * num_words + w]
        mean = count / len(participants)
        new_data[w, :] = mean
        new_y[w] = -1 if np.isnan(mean).any() else w
    new_data = new_data[new_y != -1, :]
    new_y = new_y[new_y != -1]
    return new_data, new_y, np.ones(new_y.shape[0]) * -1, np.copy(new_y)

################################ Sliding Window ################################
# start time, end time: int, in ms
# window_lengths: a list, consisting the lengths of the windows we want to try out
#   For example, if we want to slide the df with lengths 300, 400, We should pass in [300, 400]
# step_length: the distance between each window
#   For example, if step_length = 100, window_length = 300, windows are 0-300, 100-400, 200-500, etc. 
#                if step_length = 200, window_length = 300, windows are 0-300, 200-500, 400-700, etc.
# hasPreWindow: boolean, true if want to include the 200ms pre window, false otherwise
# Return: 2D list, a list containing lists of slided windows for each element in window_lengths

def slide_df(df, start_time, end_time, window_lengths, step_length):
    window_list = []
    num_win = []
    
    df = df[(df.Time >= start_time) & (df.Time < end_time)]

    for length_each_window in window_lengths:
        temp_df_list = []
        if start_time == (end_time+step_length-length_each_window):
            temp_df = df
            temp_df_list.append(temp_df)
            num_win.append(1)
        else:
            for i in range(start_time, (end_time+step_length)-length_each_window, step_length):
                temp_df = df[(df.Time < i + length_each_window) & (df.Time >= i)]
                assert len(temp_df) % length_each_window == 0
                temp_df_list.append(temp_df)
            temp_num_win = int((((end_time - start_time)-length_each_window) / step_length) + 1)
            assert len(temp_df_list) == temp_num_win
            num_win.append(temp_num_win)
        window_list.append(temp_df_list)
        
    assert len(window_list) == len(window_lengths)
    return window_list, num_win

################################ Restructure Matrix ################################
def permutation_and_average(df, avg_trial):
    grouped = df.groupby(['label'])
    shuffled_dfs = []
    for i in range(len(word_list)):
        # df_ith has all trials with word i 
        df_ith = grouped.get_group(i)
        # randomize the rows
        df_ith = df_ith.sample(frac=1).reset_index(drop=True)
        
        orig_df_len = len(df_ith)
        expected_df_len = 0
                
        # The leftover trials
        leftover = len(df_ith) % avg_trial
        halfsize_set = math.ceil(avg_trial/2)
        
        if orig_df_len < avg_trial:
            expected_df_len = math.ceil(orig_df_len/avg_trial)
            df_ith = df_ith.set_index(np.arange(len(df_ith)) // len(df_ith)).mean(level=0)
        
        # If the number of leftover trials is less than half size of a group, then we put them into the last group  
        elif leftover < halfsize_set:
            # Expected df_len after averaging should be floor(orig_df_len/avg_trial)
            expected_df_len = orig_df_len // avg_trial

            last_group_df = df_ith.tail(leftover+avg_trial)
            last_group_mean = last_group_df.mean(axis=0)

            # All other rows except for ones in the last group 
            df_ith_temp = df_ith.head(-(leftover+avg_trial))
            df_ith = df_ith_temp.set_index(np.arange(len(df_ith_temp)) // avg_trial).mean(level=0)

            # Append everything else with the last group
            df_ith = df_ith.append(last_group_mean, ignore_index=True)
        else: # Else if the number of leftover trials is more than half of a group, they form a group themselves
            # Expected df_len after averaging should be ceil(orig_df_len/avg_trial)
            expected_df_len = math.ceil(orig_df_len/avg_trial)

            df_ith = df_ith.set_index(np.arange(len(df_ith)) // avg_trial).mean(level=0)
            
        assert len(df_ith) == expected_df_len
        shuffled_dfs += [df_ith]
        
    # Append shuffled dataframes of each word together
    result = pd.concat(shuffled_dfs)
    return result