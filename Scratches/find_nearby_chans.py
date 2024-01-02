import mne
import numpy as np
import pandas as pd


# Load the eeglab.set files.
# participants = ["904", "905", "906", "908", "909", "910", "912", "913", "914", "916", "917", "919", "920",
#                 "921", "923", "924", "927", "929", "928", "930", "932"]

participants = ["105", "106", "107", "109", "111", "112", "115", "116", "117", "119", "120", "121", "122",
                "124"]

channel_loc_path = "/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/"

participant_neighbours = []


ch_group_lengths = []

for participant in participants:

    # Read the loc files.
    raw = mne.io.read_raw_eeglab(channel_loc_path + participant + ".set")
    # Set the montage.

    # raw.set_montage("GSN-HydroCel-65_1.0")
    # info = mne.io.read_info(channel_loc_path + participant + ".fif")
    # Drop channels 61 to 65.
    raw = raw.drop_channels(["E61", "E62", "E63", "E64", "E65"])
    print(raw.info["ch_names"])

    # fig = raw.plot_sensors(show_names=True)


    # Find the channel adjacencies.
    ch_adjacency, ch_names = mne.channels.find_ch_adjacency(raw.info, ch_type="eeg")
    ch_suffixes = []
    participant_neighbours_len = []
    for row_chan, col_num in enumerate(ch_adjacency.indptr[:-1]):
        neighbours = ch_adjacency.indices[col_num:ch_adjacency.indptr[row_chan + 1]]
        ch_suffixes.append([ch_names[i] for i in neighbours])
        participant_neighbours_len.append(len(neighbours))
    # participant_neighbours[participant] = ch_suffixes
    ch_group_lengths.append(participant_neighbours_len)
    participant_neighbours.append(ch_suffixes)


# NOTE: I'm only saving the first one because they're all the same (same EEG cap and montage).
np.savez_compressed("/Users/simpleparadox/PycharmProjects/jwlab_eeg/Scratches/channel_neighbours_12m.npz", participant_neighbours=participant_neighbours[0], dtype=object)

print(participant_neighbours)




# Quick check to make sure all the groups for a channel have the same size and also across participants.
sample_lengths = ch_group_lengths[0]

# Check against every row of the ch_group_lengths.
i = 0
for row in ch_group_lengths[1:]:
    if row == sample_lengths:
        print("Equal")
        i += 1
    else:
        raise ValueError("Not equal")

print(i)







