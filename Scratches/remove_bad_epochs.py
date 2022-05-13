import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne



#
participants = ["904", "905", "906", "908", "909", "910", "912", "913", "914", "916", "917", "919", "920", "921",
                         "923", "924", "927", "928", "929", "930", "932"]

# participants = ["105", "106", "107", "109", "111", "112", "115", "116", "117", "119", "120", "121", "122", "124"]

# participants = [908] # For testing only

for participant in participants:

    print(f"Participant: {participant}")

    # participant = 105
    # Read the ml_csv file (the one that is cleaned).
    ml_csv = pd.read_csv(f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/cleaned_ml_no_reref_low_pass/{participant}_cleaned_ml_no_reref_low_pass.csv')

    # Read in the labels.txt file.
    # labels = pd.read_csv(f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/label_jennlocal/{participant}_labels.txt', sep=' ', header=None).iloc[:, :].values

    labels = np.loadtxt(f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/label_jennlocal/{participant}_labels.txt')
    # Read in the obs.csv file.
    trial_cell_obs_df = pd.read_csv(f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/db_jennlocal/{participant}_trial_cell_obs.csv')
    trial_cell_obs = trial_cell_obs_df.iloc[:, :].values
    trial_cell_obs_header = trial_cell_obs_df.columns

    # Write a script that looks at the epoched data and removes bad epochs, and simlutaneously rows from 'labels' and 'trial_cell_obs'.
    # Using method from Foster et. al. 2021. If the absolute value of any epoch is greater a range then mark for removal.

    start = 0
    epoch_end = len(ml_csv)
    step = 1200
    abs_threshold = 200
    print("Original labels: ", len(labels[labels!=-1]))
    print("Original trial cell obs", len(trial_cell_obs[trial_cell_obs[:,1]!=-1]))
    epoch_indices = [epoch_index for epoch_index in range(start, epoch_end, step)]

    store_array = np.empty((1,60))  # Create an empty array. Later, I'll remove the first row.
    store_array_times = []
    store_array_labels = []
    store_array_trial_cell_obs = []

    bad_epoch_df_list = []
    good_epoch_df_list = []
    # labels = labels.tolist()


    for label_index, start_index in enumerate(epoch_indices):
        epoch_df_times = ml_csv.iloc[start_index:start_index+step,0].values.tolist()
        epoch_df = ml_csv.iloc[start_index:start_index + step, 1:]
        epoch_df_values = ml_csv.iloc[start_index:start_index + step, 1:].values

        flag = True
        if int(labels[label_index]) == -1:
            flag = False

        # Now remove the rows for which any electrodes cross a certain absolute value.
        if flag:
            for electrode in epoch_df.columns:
                electrode_values = epoch_df[electrode].values
                if abs(max(electrode_values) - min(electrode_values)) > abs_threshold:
                    # Remove the epoch.
                    # print(f"Epoch {label_index}")
                    # print(f"Electrode {electrode}")
                    # print(abs(max(electrode_values) - min(electrode_values)))
                    labels[label_index] = -1
                    trial_cell_obs[label_index, 1] = -1
                    flag = False
                    break

        if flag:
            store_array = np.append(store_array, epoch_df_values, axis=0)
            store_array_times.extend(epoch_df_times)
            # assert labels[label_index] != -1
            store_array_labels.append(labels[label_index])
            store_array_trial_cell_obs.append(trial_cell_obs[label_index,:].tolist())

            # Store the epoch df for later concatenation.
            temp = epoch_df.iloc[:,:].values
            good_epoch_df_list.append(temp)
            # bad_epoch_df_list.append(np.zeros_like(temp))
        else:
            temp = epoch_df.iloc[:, :].values
            bad_epoch_df_list.append(temp)
            # good_epoch_df_list.append(np.zeros_like(temp))




    # Save the trial_cell_obs, and labels.
    # print("Reduced labels: ", len(labels[labels != -1]))
    # print("Reduced trial cell obs", len(trial_cell_obs[trial_cell_obs[:, 1] != -1]))
    print("Reduced labels: ", len(labels[labels!=-1]))
    print("Reduced trial cell obs", len(trial_cell_obs[trial_cell_obs!=-1]))
    np.savetxt(f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/label_abs_low_pass_after_bad_remove/{participant}_labels.txt', np.array(labels.tolist(), dtype=np.int32).tolist(), delimiter=' ', fmt='%d')



    # TODO: Use MNE to create a raw object (using np.txt(path_to_csv)) and then do the average referencing in MNE.
    # Then save it as a CSV. Then do the decoding analysis.

    # Here's the roadmap to remove the trials and then rereference the data.
    # Remove bad trials from the ml_csv file and also simultaneously remove them the indices from the labels.txt file.
    # Then create an mne object and then do averaging referencing.

    # data = ml_csv.iloc[:, :].values
    # data = np.loadtxt(f"/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/cleaned2/{participant}_cleaned_ml.csv")
    ch_names = ml_csv.columns.values
    ch_names = np.delete(ch_names, [0]).tolist() # Removing the time channel
    sfreq = 1000
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    store_array = store_array[1:,:]  # Removing the first because it was dummy data.
    store_array = np.transpose(store_array)
    raw = mne.io.RawArray(store_array, info)
    raw_avg = raw.set_eeg_reference(ref_channels=ch_names)

    raw_avg_data = np.transpose(raw_avg[:,:][0])

    # raw_avg_data = np.insert(raw_avg_data, 0, store_array_times, axis=1)


    # Let's add the old data as well which wasn't used to do the average referencing. This will help keep things consistent on compute canada as well (with ML-badTrials 2.xlxs file).
    final_data = np.empty((1,60))
    good_idx = 0
    bad_idx = 0
    start = 0
    step = 1200
    good_epoch_indices = [epoch_index for epoch_index in range(start, len(raw_avg_data), step)]
    # epoch data again
    raw_epoched_data = []
    for epoch_idx, start_idx in enumerate(good_epoch_indices):
        raw_epoched_data.append(raw_avg_data[start_idx:start_idx+step, :])


    for label_idx, label in enumerate(labels):
        # print(label_idx, " ", label)
        if int(label) == -1:
            # print("Bad idx: ", bad_idx)
            t = bad_epoch_df_list[bad_idx]
            final_data = np.concatenate((final_data, t), axis=0)
            bad_idx += 1
        else:
            # print("Good idx: ", good_idx)
            t = raw_epoched_data[good_idx]
            final_data = np.concatenate((final_data, t), axis=0)
            good_idx += 1


    final_data = final_data[1:]
    all_times = ml_csv.iloc[:,0].values
    final_data = np.insert(final_data, 0, all_times, axis=1)





    # Now store the processed mne object to a csv file.
    raw_processed_columns = ['Time']
    raw_processed_columns.extend(ch_names)
    raw_processed = pd.DataFrame(final_data, columns=raw_processed_columns)
    raw_processed.to_csv(f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/cleaned_ml_no_reref_low_pass_after_bad_remove/{participant}_cleaned_ml.csv', index=False)

    trial_cell_obs_mod_df = pd.DataFrame(trial_cell_obs, columns=trial_cell_obs_header)
    trial_cell_obs_mod_df.to_csv(f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/db_abs_low_pass_after_bad_remove/{participant}_trial_cell_obs.csv', index=False)
    # print("Reduced labels: ", len(labels[labels!=-1]))
    # print("Reduced trial cell obs", len(trial_cell_obs[trial_cell_obs[:,1]!=-1]))
