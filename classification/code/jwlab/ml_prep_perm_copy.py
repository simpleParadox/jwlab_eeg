import math
import random
from numpy import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
from IPython.display import display
from jwlab.data_graph import plot_good_trial_participant, plot_good_trial_word
from jwlab.participants_map import map_participants
from jwlab.bad_trials import get_bad_trials, get_left_trial_each_word
from jwlab.constants import word_list, bad_trials_filepath, old_participants, cleaned_data_filepath, cleaned_data_filepath_new_filter, cleaned_data_filepath_ref_e65, \
                    cleaned_data_filepath_003highpass_avg_ref, cleaned_ml_mar22_band_rmbase_basline_corr, cleaned_ml_reject_artifact_asr, cleaned_data_filepath_reref_band_after_bad_remove, \
                    cleaned_data_filepath_reref_low_pass_after_bad_remove, cleaned_data_filepath_linear_detrend_band_no_bad_remove, cleaned_pic_band_filepath, \
                    cleaned_data_filepath_linear_detrend_low_no_bad_remove, adam_detrend_low_no_bad_remove_filepath, label_filepath, adam_detrend_low_no_bad_rm_no_base_filepath, \
                    adam_detrend_low_no_bad_rm_40_order_filepath
# from jwlab.constants import word_list, bad_trials_filepath, old_participants, cleaned_data_filepath, label_filepath





labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}



################################ prep data ################################

def init(age_group, selected_ps):
    if age_group == 9:
        if selected_ps == False:
            participants = ["904", "905" ,"906", "908", "909", "910", "912", "913", "914", "916", "917", "919", "920", "921",
                         "923", "924", "927", "929", "928", "930", "932"]
        elif selected_ps:
            participants = ["904", "905", "908", "910", "912", "913", "921", "923", "927", "929", "930"]

    # all
    #         participants = [ "904", "905","906", "908", "909", "912", "913", "914", "916", "917", "919", "920", "921", "923", "924", "927", "929","928", "930", "932"]

    elif age_group == 12:
        if selected_ps == False:
            participants = ["105", "106", "107", "109", "111", "112", "115", "116", "117", "119", "120", "121", "122", "124"]
        elif selected_ps:
            participants = ["105", "106", "107", "109", "112", "115", "116", "117", "120", "121", "122", "124"]
    else:
        raise ValueError("Unsupported age group!")
    
    print("Selected Participants: ", selected_ps)
    print("Selected participants: ", participants)

    return participants


def load_ml_data(participants):
    # read all participant csvs, concat them into one dataframe
    # Using the data where E65 was used as a reference instead of average reference.
    # print(cleaned_ml_reject_artifact_asr)
    # dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (adam_detrend_low_no_bad_remove_filepath, s))
    #        for s in participants]
    # Uncomment this for no baseline correction and no high-filtering.
    dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (cleaned_data_filepath, s))
           for s in participants]
    
    df = pd.concat(dfs, axis=0, ignore_index=True, sort=True)
    try:
        df = df.drop('E65', axis=1)
    except Exception as e:
        print(e)
    

    jenn_local_label_filepath = label_filepath  # "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/label_jennlocal/"
    # jenn_local_label_filepath = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/label_abs_remove_200uv/"
    # jenn_local_label_filepath = "/home/rsaha/scratch/jwlab_eeg/data/label_abs_remove_reref_band_after_bad_remove/"
    # jenn_local_label_filepath = "/home/rsaha/scratch/jwlab_eeg/data/label_abs_low_pass_after_bad_remove/"
    ys = [np.loadtxt("%s%s_labels.txt" % (jenn_local_label_filepath, s)).tolist() for s in participants]

    
    # Change the following boolean value to True or False.
    # detrending = False
    # if detrending == False:
    #     # Write code here for the scaling operation only.
    #     print("No detrending")
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df.iloc[:,:-1].values) # Excluding the 'Time' column.
    new_df = pd.DataFrame(scaled_df, index=df.index, columns=df.columns[:-1])
    df = pd.concat([new_df, df['Time']], axis=1)
    return df, ys
    # else:
    #     # 'detrending = True.
    #     # The following code section is for masked-trial robust detrending.
    #     #----------------------------------------
    #     # Change parameters here.
    #     scaling_first = False
    #     pre_onset = True # doesn't really matter when masked_trial_detrending is 'False'.
    #     masked_trial_detrending = False # Set this to 'False' for regular robust detrending.

    #     # -------------------------------------------------------------------
    #     if scaling_first == True:
    #         scaler = StandardScaler()
    #         scaled_df = scaler.fit_transform(df.iloc[:, :-1].values)
    #         new_df = pd.DataFrame(scaled_df, index=df.index, columns=df.columns[:-1])

    #         if pre_onset == False:
    #             # Setting post-onset weights to zero.
    #             events = np.arange(0, len(new_df), 1200)  # np.arange(0, len(new_df), 1200)
    #             tmin = 0.2  # 0
    #             tmax = 1.2  # 0.2
    #         else:
    #             # Setting pre-onset weights to zero
    #             events = np.arange(0, len(new_df), 1200)
    #             tmin = 0
    #             tmax = 0.2
    #         sfreq = 1000

    #         detrend_weights = None
    #         if masked_trial_detrending == False:
    #             print('Regular Robust detrending')
    #             y, w, r = detrend(new_df.to_numpy(), order=2, w=None)
    #         else:
    #             print('Masked Trial Robust detrending scaling first')
    #             detrend_weights = create_masked_weight(new_df, events, tmin, tmax, sfreq)
    #             y, w, r = detrend(new_df.to_numpy(), order=2, w=detrend_weights)

    #         # Add the 'Time' column back.
    #         df_with_time = pd.concat([pd.DataFrame(y, index=df.index, columns=df.columns[:-1]), df['Time']], axis=1)
    #         return df_with_time, ys
    #     else:
    #         # scaling_first = False
    #         df_no_time = df.drop('Time', axis=1)
    #         if pre_onset == False:
    #             # Setting post-onset weights to zero.
    #             events = np.arange(0, len(df_no_time), 1200)  # np.arange(0, len(new_df), 1200)
    #             tmin = 0.2  # 0
    #             tmax = 1.2  # 0.2
    #         else:
    #             # Setting pre-onset weights to zero
    #             events = np.arange(0, len(df_no_time), 1200)
    #             tmin = 0
    #             tmax = 0.2
    #         sfreq = 1000

    #         detrend_weights = None
    #         if masked_trial_detrending == False:
    #             print('Regular robust detrending')
    #             y, w, r = detrend(df_no_time.to_numpy(), order=30, w=None)
    #         else:
    #             print('Masked-trial robust detrending scaling later')
    #             detrend_weights = create_masked_weight(df_no_time, events, tmin, tmax, sfreq)
    #             y, w, r = detrend(df_no_time.to_numpy(), order=20, w=detrend_weights)
    #         scaler = StandardScaler()
    #         scaled_df = scaler.fit_transform(y)  # .iloc[:, :-1].values)
    #         new_df = pd.DataFrame(scaled_df, index=df.index, columns=df.columns[:-1])
    #         df_with_time = pd.concat([new_df, df['Time']], axis=1)
    #         return df_with_time, ys
    # #----------------------------------------


def prep_ml(age_group, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000, selected_ps=False, pre_standard=False):
    participants = init(age_group, selected_ps)
    df, ys = load_ml_data(participants)
    # if not pre_standard:
    return prep_ml_internal(df, ys, participants, useRandomizedLabel, averaging, sliding_window_config,
                        downsample_num=downsample_num, selected_ps=selected_ps, test_size=0.2)
    # else:
    #     return df, ys




# This function is not being used now.
def pre_window_slicing(df, ys, age_group, useRandomizedLabel, averaging, sliding_window_config, test_size=0.2, selected_ps=False):
    print("Inside pre_window_slicing")
    participants = init(age_group, selected_ps)
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
            ys[each_ps][ybad[each_ps][bad_trial] - 1] = -1

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
        np.random.shuffle(Y)
        random.shuffle(Y)

    # Assign trial numbers here.
    y_repeat = np.repeat(Y.astype(int), 1200).astype(int)
    y_trial = np.repeat(np.arange(len(Y)), 1200).astype(int)
    df['label'] = y_repeat
    df['trial'] = y_trial
    participant_list = np.concatenate([[ys.index(y)] * len(y) for y in ys])
    participant_list_repeat = np.repeat(participant_list, 1200).astype(int)
    df['participant'] = participant_list_repeat

    # Get only the good indices. Cuz we don't want to consider bad trials for data scaling.
    df = df[df['label']!=-1]


    # Use label_repeat to obtain the test_indices.
    label_repeat_set = set(df['trial'].values.tolist())
    num_indices = len(label_repeat_set)
    fivefold_testsize = int(test_size * num_indices)
    # Now randomly select test_indices.
    test_indices = np.random.choice(num_indices-1, fivefold_testsize, replace=False)

    test_df = df[df['trial'].isin(test_indices)]
    train_df = df[~df['trial'].isin(test_indices)]

    scaler = StandardScaler()
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    train_df_copy.iloc[:, :-4] = scaler.fit_transform(train_df.iloc[:, :-4])
    test_df_copy.iloc[:, :-4] = scaler.transform(test_df.iloc[:, :-4])


    # Now merge and sort the scaled dataframes.
    merged_df = pd.concat([train_df_copy, test_df_copy])
    sorted_merged_df = merged_df.sort_index()

    #### Sliding window section ####
    start_time = sliding_window_config[0]
    end_time = sliding_window_config[1]
    window_lengths = sliding_window_config[2]
    step_length = sliding_window_config[3]

    windows_list, num_win = slide_df(sorted_merged_df, start_time, end_time, window_lengths, step_length)

    X_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]
    y_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]
    p_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]
    w_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]
    df_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]

    # First remove all the bad trials (denoted as -1) from all the lists in ys.
    temp_ys = []
    for ys_list in ys:
        t = np.array(ys_list)
        temp_ys.append(t[t!=-1].tolist())





    for length_per_window in range(len(windows_list)):
        for each_window in range(len(windows_list[length_per_window])):
            df = windows_list[length_per_window][each_window]
            try:
                df = df.drop(columns=["Time", 'trial'], axis=1)
            except:
                print("Time not found.")
            # X = df.values

            # Obtain the labels here and make them unique.
            # Obtain the participants column here and make them unique.
            # TODO: But keep them in order.
            labels = df['label']
            labels = [labels.values.tolist()[e] for e in range(0, len(labels), 100)]  # Retrieve unique index values.



            participants_column = np.concatenate([[temp_ys.index(ty)] * len(ty) for ty in temp_ys])
            X = df.iloc[:, :-2].values
            X = np.reshape(X, (window_lengths[length_per_window], 60, -1))
            (i, j, k) = X.shape
            X = np.reshape(X, (k, j * window_lengths[length_per_window]))

            # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
            df = pd.DataFrame(data=X)
            df['label'] = labels
            df['participant'] = participants_column

            # remove bad samples
            # df = df[df.label != -1]

            # make label zero indexed
            df.label -= 1

            # different averaging processes
            if averaging == "no_averaging":
                X, y, p, w = no_average(df)
            elif averaging == "average_trials":
                X, y, p, w = average_trials(df)
            elif averaging == "average_trials_and_participants":
                X, y, p, w = average_trials_and_participants(df, participants)
            elif averaging == "no_average_labels":
                X, y, p, w = no_average_labels(df)
            elif averaging == "permutation":
                ## change below to change the averaging set size
                df = permutation_and_average(df, 5)
                # X, y, p, w = no_average(df)  # Changing this to no_average_labels in the next line.
                X, y, p, w = no_average_labels(df)  # using this to averaging the test data.
            elif averaging == "permutation_with_labels":
                ## change below to change the averaging set size
                df = permutation_and_average(df, 20)
                X, y, p, w = no_average_labels(df)
            else:
                raise ValueError("Unsupported averaging!")
            
            try:
                y = y.tolist()
            except:
                print("y already in list form")

            X_list[length_per_window][each_window] = X
            y_list[length_per_window][each_window] = y
            p_list[length_per_window][each_window] = p
            w_list[length_per_window][each_window] = w
            df_list[length_per_window][each_window] = df

    return X_list, y_list, test_indices, [good_trial_participant_count, good_trial_word_count], num_win


def prep_ml_internal(df, ys, participants, useRandomizedLabel, averaging, sliding_window_config, downsample_num=1000, selected_ps=False, test_size=0.2):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    # df = df[df.Time >= 0]
    print("Inside prep_ml_internal")

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
            ys[each_ps][ybad[each_ps][bad_trial] - 1] = -1

        # count the total number of trials for each participant
        trial_count += [len(ys[each_ps])]
        bad_trial_count += [len(ybad[each_ps])]

    # good trial each participant
    good_trial_participant_count = np.around(np.true_divide(
        np.subtract(trial_count, bad_trial_count), trial_count), decimals=2)
    # good trial each word each participant
    good_trial_word_count = get_left_trial_each_word(participants)

    Y = np.concatenate(ys)
    # print("Y is",Y)

    if useRandomizedLabel:
        np.random.shuffle(Y)
        random.shuffle(Y)
        # remap_label(Y)  ## Changed from the above two lines to this -> just to test it out. Commenting this out and using the later one.
        # print("Labels shuffled.")

    #### Sliding window section ####
    start_time = sliding_window_config[0]
    end_time = sliding_window_config[1]
    window_lengths = sliding_window_config[2]
    step_length = sliding_window_config[3]

    # Do the scaling on all the data first and then do slide df. Only fit the scaler on the training set and use that to transform the test set.

    windows_list, num_win = slide_df(df, start_time, end_time, window_lengths, step_length)

    X_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]
    y_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]
    p_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]
    w_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]
    df_list = [[0 for j in range(num_win[i])] for i in range(len(window_lengths))]

    for length_per_window in range(len(windows_list)):
        for each_window in range(len(windows_list[length_per_window])):
            df = windows_list[length_per_window][each_window]
            df = df.drop(columns=["Time"], axis=1)
            X = df.values
            X = np.reshape(
                X, (window_lengths[length_per_window], 60, -1))
            # X = resample(X, downsample_num, axis=0)
            (i, j, k) = X.shape
            X = np.reshape(X, (k, j * window_lengths[length_per_window]))

            # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
            df = pd.DataFrame(data=X)
            df['label'] = Y
            df['participant'] = np.concatenate(
                [[ys.index(y)] * len(y) for y in ys])

            # remove bad samples
            df = df[df.label != -1]

            # make label zero indexed
            df.label -= 1
            # print(max(df.label))

            # different averaging processes
            if averaging == "no_averaging":
                X, y, p, w = no_average(df)
            elif averaging == "average_trials":
                X, y, p, w = average_trials(df)
            elif averaging == "average_trials_and_participants":
                X, y, p, w = average_trials_and_participants(df, participants)
            elif averaging == "no_average_labels":
                X, y, p, w = no_average_labels(df)
            elif averaging == "permutation":
                ## change below to change the averaging set size
                df = permutation_and_average(df, 5)
                # X, y, p, w = no_average(df)  # Changing this to no_average_labels in the next line.
                X, y, p, w = no_average_labels(df)  # using this to averaging the test data.
            elif averaging == "permutation_with_labels":
                ## change below to change the averaging set size
                df = permutation_and_average(df, 20)
                X, y, p, w = no_average_labels(df)
            else:
                raise ValueError("Unsupported averaging!")

            # if useRandomizedLabel:
            #     y = remap_label(y)
            # #     random.shuffle(y)
            # #     np.random.shuffle(y)

            ## binary: animacy
            # y[y < 8] = 0
            # y[y >= 8] = 1

            #     #mom and baby vs all
            #             y[y == 0] = 0
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

            #             #words starting with b

            #             y[y == 0] = 0
            #             y[y == 1] = 0
            #             y[y == 2] = 0
            #             y[y == 3] = 0
            #             y[y == 4] = 1
            #             y[y == 5] = 1
            #             y[y == 6] = 1
            #             y[y == 7] = 1

            #             y[y == 8] = 0
            #             y[y == 9] = 0
            #             y[y == 10] = 1
            #             y[y == 11] = 1
            #             y[y == 12] = 1
            #             y[y == 13] = 1
            #             y[y == 14] = 1
            #             y[y == 15] = 1

            X_list[length_per_window][each_window] = X
            y_list[length_per_window][each_window] = y
            p_list[length_per_window][each_window] = p
            w_list[length_per_window][each_window] = w
            df_list[length_per_window][each_window] = df

    return X_list, y_list, [good_trial_participant_count, good_trial_word_count], num_win



def remove_samples(X):

    labels = X[0][0]['label'].values
    df_index = X[0][0].index
    indices = []
    for lab in labels_mapping.keys():
        idxs = [df_index[i] for i, x in enumerate(labels) if x == int(lab)]
        indices.append(idxs)

    # Now randomly choose the elements to be removed.
    indices_to_drop = []
    for idx in indices:
        # idx is a list.
        no_elements_to_delete = 8 #len(idx) // 4
        no_elements_to_keep = len(idx) - no_elements_to_delete
        b = set(random.sample(idx, no_elements_to_delete))  # the `if i in b` on the next line would benefit from b being a set for large lists
        b = [i for i in idx if i in b]  # you need this to restore the order
        # Use b to drop the samples from X - for each window.
        indices_to_drop.append(b) # Length of indices_to_drop should be 16.

    idxs_to_drop = np.array(indices_to_drop)
    idxs_to_drop = idxs_to_drop.flatten()
    # Now drop the indices from the dataframes.
    dfs = []
    for i in range(len(X)):
        for j in range(len(X[i])):
            dfs.append(X[i][j].drop(idxs_to_drop, axis=0))

    return [dfs]

def prep_matrices_avg(X, age_group, useRandomizedLabel, train_only=False, test_size=0.20, selected_ps=False, test_idxs=None):
    """
    Randomly select indices from the dataset to make training and test data.
    """
    print("Inside prep_matrices_avg")
    participants = init(age_group, selected_ps=selected_ps)
    num_indices = len(X[0][0])
    fivefold_testsize = int(test_size * num_indices)
    # if type(test_idxs) == list:
    #     test_indices = test_idxs
    # else:
    test_indices = np.random.choice(num_indices - 1, fivefold_testsize, replace=False)
    print("Assigned test_indices")


    df_test_m = []
    df_train_m = []
    for i in range(len(X)):
        df_test = []
        df_train = []
        for j in range(len(X[0])):
            ## will need each window
            try:
                # This is because I'm loading the dataset only once.
                X[i][j] = X[i][j].reset_index()
            except ValueError:
                print("reset_index already done", end=' ')

            # #create new df with these indices and removing from orig
            df_test.append(X[i][j].iloc[test_indices])
            df_train.append(X[i][j].drop(X[i][j].index[test_indices]))
            assert (len(df_train[i][j]) + len(df_test[i][j]) == len(X[i][j]))
            df_test[j] = df_test[j].drop(columns=['index'], axis=1)
            df_train[j] = df_train[j].drop(columns=['index'], axis=1)
        df_test_m.append(df_test)
        df_train_m.append(df_train)

    X_train = []
    y_train = []

    for i in range(len(X)):
        # create training matrix:
        X_train_i = []
        y_train_i = []
        for j in range(len(X[0])):
            y_train_i.append(df_train_m[i][j].label.values)
            # if use_randomized_label:
            #     np.random.shuffle(y_train_i)
            #     random.shuffle(y_train_i)
            X_train_i.append(df_train_m[i][j].drop(columns=['label', 'participant'], axis=1))
        X_train.append(X_train_i)
        y_train.append(y_train_i)
        # if use_randomized_label:
        #     np.random.shuffle(y_train)
        #     random.shuffle(y_train)


    if train_only == True:
        return X_train, None, y_train, None
    # create test matrices
    X_test = []  # test raw trials
    y_test = []
    X_test_pt = []  # test avg trials and ps
    y_test_pt = []

    for i in range(len(X)):
        X_test_i = []
        y_test_i = []
        for j in range(len(X[0])):
            X_test_pt_temp, y_test_temp_pt, ps, w = average_trials_and_participants(df_test_m[i][j], participants)
            X_test_i.append(pd.DataFrame(X_test_pt_temp))
            # if use_randomized_label:
            #     np.random.shuffle(y_test_temp_pt)
            #     random.shuffle(y_test_temp_pt)
            y_test_i.append(y_test_temp_pt)

        X_test_pt.append(X_test_i)
        y_test_pt.append(y_test_i)
        # if use_randomized_label:
        #     np.random.shuffle(y_test_pt)
        #     random.shuffle(y_test_pt)

    # binary classification, comment these if you want the labels only. Commented out by Rohan.
    # for i in range(len(X)):
    #     for j in range(len(X[0])):
    #         y_train[i][j][y_train[i][j] < 8] = 0
    #         y_train[i][j][y_train[i][j] >= 8] = 1
    #
    #         y_test_pt[i][j][y_test_pt[i][j] < 8] = 0
    #         y_test_pt[i][j][y_test_pt[i][j] >= 8] = 1

    return X_train, X_test_pt, y_train, y_test_pt



# Raw data
def no_average(df):
    return df.drop(columns=['label', 'participant'], axis=1).values, df.label.values.flatten(), df.participant.values, df.label.values


def no_average_labels(df):
    return df, df.label.values.flatten(), df.participant.values, df.label.values


# For each participant, average the value for each word. Expected shape[0] is len(participants) x len(word_list)
def average_trials(df):
    num_participants = int(df.participant.max() + 1)
    num_words = len(word_list)

    new_data = np.zeros((num_participants * num_words, len(df.columns) - 2))
    df_data = df.drop(columns=['label', 'participant'], axis=1)
    new_y = np.zeros(num_participants * num_words)
    participants = np.zeros(num_participants * num_words)

    for p in range(num_participants):
        for w in range(num_words):
            means = df_data[np.logical_and(df.participant == p, df.label == w)].values.mean(axis=0
                                                                                            ) if df_data[np.logical_and(
                df.participant == p, df.label == w)].size != 0 else 0
            new_data[p * num_words + w, :] = means
            new_y[p * num_words + w] = -1 if np.isnan(means).any() else w
            participants[p * num_words + w] = p
    return new_data, new_y, participants, np.copy(new_y)

minimal_mouth_labels_exclude = {4: 'cat', 5: 'dog', 7: 'mom',
                  8: 'banana', 9: 'bottle', 11: 'cracker'}

# Use this when training only on non-minimal mouth information words. These are the words which will be excluded.
minimal_mouth_labels_include = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                    6: 'duck', 10: 'cookie', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}
# Average the value of each word across participants. Expected shape[0] is len(word_list)
def average_trials_and_participants(df, participants):
    num_words = len(word_list)
    
    # num_words = len(minimal_mouth_labels_exclude)
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
    # print("avg_tri_and_ps func compl")
    return new_data, new_y, np.ones(new_y.shape[0]) * -1, np.copy(new_y)


def remap_label(y):
    print("Remap label")
    labels_temp = list(range(0, 16))
    random.shuffle(labels_temp)

    mapdict = {}
    for i in range(16):
        mapdict[i] = labels_temp.index(i)

    newArray = copy(y)

    for k, v in mapdict.items():
        newArray[y == k] = v

    # np.random.shuffle(newArray)
    return newArray


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
    print("Inside slide_df")
    window_list = []
    num_win = []
    df['Time'] = df['Time'].astype(int)
    df = df[(df.Time >= start_time) & (df.Time < end_time)]


    

    for length_each_window in window_lengths:
        temp_df_list = []
        if start_time == (end_time + step_length - length_each_window):
            temp_df = df
            temp_df_list.append(temp_df)
            num_win.append(1)
        else:
            for i in range(start_time, (end_time + step_length) - length_each_window, step_length):
                temp_df = df[(df.Time < i + length_each_window) & (df.Time >= i)]
                assert len(temp_df) % length_each_window == 0
                temp_df_list.append(temp_df)
            temp_num_win = int((((end_time - start_time) - length_each_window) / step_length) + 1)
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
        halfsize_set = math.ceil(avg_trial / 2)

        if orig_df_len < avg_trial:
            expected_df_len = math.ceil(orig_df_len / avg_trial)
            df_ith = df_ith.set_index(np.arange(len(df_ith)) // len(df_ith)).mean(level=0)

        # If the number of leftover trials is less than half size of a group, then we put them into the last group
        elif leftover < halfsize_set:
            # Expected df_len after averaging should be floor(orig_df_len/avg_trial)
            expected_df_len = orig_df_len // avg_trial

            last_group_df = df_ith.tail(leftover + avg_trial)
            last_group_mean = last_group_df.mean(axis=0)

            # All other rows except for ones in the last group
            df_ith_temp = df_ith.head(-(leftover + avg_trial))
            df_ith = df_ith_temp.set_index(np.arange(len(df_ith_temp)) // avg_trial).mean(level=0)

            # Append everything else with the last group
            df_ith = df_ith.append(last_group_mean, ignore_index=True)
        else:  # Else if the number of leftover trials is more than half of a group, they form a group themselves
            # Expected df_len after averaging should be ceil(orig_df_len/avg_trial)
            expected_df_len = math.ceil(orig_df_len / avg_trial)

            df_ith = df_ith.set_index(np.arange(len(df_ith)) // avg_trial).mean(level=0)

        assert len(df_ith) == expected_df_len
        shuffled_dfs += [df_ith]

    # Append shuffled dataframes of each word together
    result = pd.concat(shuffled_dfs)
    return result