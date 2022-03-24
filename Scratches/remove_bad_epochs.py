import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# participants = ["904", "905", "906", "908", "909", "910", "912", "913", "914", "916", "917", "919", "920", "921",
#                          "923", "924", "927", "928", "929", "930", "932"]

participants = ["105", "106", "107", "109", "111", "112", "115", "116", "117", "119", "120", "121", "122", "124"]

# participants = [908] # For testing only

for participant in participants:

    print(f"Participant: {participant}")

    # participant = 105
    # Read the ml_csv file (the one that is cleaned).
    ml_csv = pd.read_csv(f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/cleaned2/{participant}_cleaned_ml.csv')

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


    for label_index, start_index in enumerate(epoch_indices):
        epoch_df = ml_csv.iloc[start_index:start_index + step, 1:-1]
        # break
        # Now remove the rows for which any electrodes cross a certain absolute value.
        for electrode in epoch_df.columns:
            electrode_values = epoch_df[electrode].values
            if abs(max(electrode_values) - min(electrode_values)) > abs_threshold:
                # Remove the epoch.
                # print(f"Epoch {label_index}")
                # print(f"Electrode {electrode}")
                # print(abs(max(electrode_values) - min(electrode_values)))
                labels[label_index] = -1
                trial_cell_obs[label_index, 1] = -1


    # labels = np.array(labels[0].tolist(), dtype=np.int32).tolist()
    # Save the trial_cell_obs, and labels.
    print("Reduced labels: ", len(labels[labels != -1]))
    print("Reduced trial cell obs", len(trial_cell_obs[trial_cell_obs[:, 1] != -1]))

    np.savetxt(f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/label_abs_remove_200uv/{participant}_labels.txt', np.array(labels.tolist(), dtype=np.int32).tolist(), delimiter=' ', fmt='%d')

    trial_cell_obs_mod_df = pd.DataFrame(trial_cell_obs, columns=trial_cell_obs_header)
    trial_cell_obs_mod_df.to_csv(f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/db_abs_remove_200uv/{participant}_trial_cell_obs.csv', index=False)


    print("Reduced labels: ", len(labels[labels!=-1]))
    print("Reduced trial cell obs", len(trial_cell_obs[trial_cell_obs[:,1]!=-1]))
