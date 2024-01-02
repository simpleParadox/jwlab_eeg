import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Read the group channel accuracy values.

age_group = 12
wind_start1 = 0
wind_end1 = 300
wind_start2 = 300
wind_end2 = 600
wind_start3 = 600
wind_end3 = 900
group_channel_results_path = "/Users/simpleparadox/PycharmProjects/jwlab_eeg/regression/group_channel_results/"

ch_group_num = 0
acc_n_channels_first = []
acc_n_channels_second = []
acc_n_channels_third = []
while ch_group_num < 60:
    data_path1 = group_channel_results_path + f'ch_group_{age_group}m_window_{wind_start1}_to_{wind_end1}_ch_group_{ch_group_num}.npz'
    data_path2 = group_channel_results_path + f'ch_group_{age_group}m_window_{wind_start2}_to_{wind_end2}_ch_group_{ch_group_num}.npz'
    data_path3 = group_channel_results_path + f'ch_group_{age_group}m_window_{wind_start3}_to_{wind_end3}_ch_group_{ch_group_num}.npz'
    try:

        # Load the data using the data_path.
        data1 = np.load(data_path1, allow_pickle=True)
        acc_n_channels_first.append(np.mean(data1['arr_0'].tolist()[0][0]))
    except:
        print(f"No data for channel {ch_group_num} for window 1")

    try:
        data2 = np.load(data_path2, allow_pickle=True)
        acc_n_channels_second.append(np.mean(data2['arr_0'].tolist()[0][0]))
    except:
        print(f"No data for channel {ch_group_num} for window 2")

    try:
        # For the third window.
        data3 = np.load(data_path3, allow_pickle=True)
        acc_n_channels_third.append(np.mean(data3['arr_0'].tolist()[0][0]))
    except:
        print(f"No data for channel {ch_group_num} for window 3")
    ch_group_num += 1


# print(acc_n_channels)


# Read the channel positions using mne.
channel_loc_path = "/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/"
if age_group == 9:
    participant = '904'
else:
    participant = '105'
raw = mne.io.read_raw_eeglab(channel_loc_path +  participant + ".set")
raw = raw.drop_channels(["E61", "E62", "E63", "E64", "E65"])

# Set the montage.
raw.set_montage("GSN-HydroCel-65_1.0")

# Plot the topomap.
plt.clf()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8))

im1, cn1 = mne.viz.plot_topomap(acc_n_channels_first, raw.info, axes=ax1, cmap='seismic', vmin=0.35, vmax=0.65, show=False)
im2, cn2 = mne.viz.plot_topomap(acc_n_channels_second, raw.info, axes=ax2, cmap='seismic', vmin=0.35, vmax=0.65, show=False)
im3, cn3 = mne.viz.plot_topomap(acc_n_channels_third, raw.info, axes=ax3, cmap='seismic', vmin=0.35, vmax=0.65, show=False)
ax_x_start = 0.95
ax_x_width = 0.02
ax_y_start = 0.25
ax_y_height = 0.5
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im3, cax=cbar_ax)
clb.ax.tick_params(labelsize=12)
clb.set_label('2 vs 2 Accuracy', fontsize=16, rotation=270, labelpad=25)

# plt.colorbar(im1, ax=ax3)

# plt.tight_layout()
ax1.set_title(f"{wind_start1} - {wind_end1} ms", fontsize=16)
ax2.set_title(f"{wind_start2} - {wind_end2} ms", fontsize=16)
ax3.set_title(f"{wind_start3} - {wind_end3} ms", fontsize=16)
# fig.suptitle(f"Electrode (group) analysis {age_group}-month-old", fontsize=12)
# plt.title(f"Electrode selection analysis", fontsize=16, loc='right')
# plt.tight_layout()
# fig.colorbar(aspect=30)
# ax1.set_aspect(1)
# ax2.set_aspect(1)
# ax3.set_aspect(1)
# plt.show()
# Save figure and make it tight.
plt.savefig(f"Topomap_{age_group}m_300ms_windows.png", bbox_inches='tight')

# plt.colorbar(im, ax=None)
# plt.title(f"Electrode (group) analysis {age_group} Months")
# plt.show()
