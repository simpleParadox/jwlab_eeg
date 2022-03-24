import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mne
import sys
sys.path.insert(1, '/Users/simpleparadox/PycharmProjects/jwlab_eeg/classification/code')  ## For loading the following files.
sys.path.insert(1, '/Users/simpleparadox/PycharmProjects/jwlab_eeg')
# from jwlab.ml_prep_perm import prep_ml, prep_matrices_avg, load_ml_data, init

# def get_data(age_group):
#     participants = init(age_group)
#     df, ys = load_ml_data(participants)
#     return df, ys
#
#
#
#
# df, ys = load_ml_data(9)
# columns = df.columns


def viz_eeg(participant):
    # participant_path = f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/cleaned_ml_mar2022_band_rmbase_baseline_corr/{participant}_cleaned_ml.csv'
    participant_path = f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/cleaned_ml_mar2022_band_rmbase_baseline_corr/{participant}_cleaned_ml.csv'
    label_path = f'/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/label_abs_remove_200uv/{participant}_labels.txt'
    labels = np.loadtxt(label_path)
    ml_csv_file = pd.read_csv(participant_path)
    interval = 1200
    df = ml_csv_file.drop(['Time'], axis=1)
    epochs = []

    epoch_starts = [i for i in range(0, len(df), interval)]

    for label_idx, i in enumerate(epoch_starts):
        # Get the epochs.
        if int(labels[label_idx]) != -1:
            epochs.append(df.iloc[i:i+interval, :].values)
    epochs = np.array(epochs)
    averaged_epochs = np.mean(epochs, axis=0)

    data = pd.DataFrame(averaged_epochs, columns=df.columns)
    plt.clf()
    sns.lineplot(data=data)
    plt.xticks([0, 200, 400, 600, 800, 1000, 1200], ['-200', '0', '200', '400', '600', '800', '1000'])
    plt.axhline(0,-200,1000)
    plt.axvline(200, -20, 20)
    plt.legend("")
    plt.xlabel('Time (ms)')
    plt.ylabel('micro-volts')
    plt.title(f'{participant} 0.1-50hz baseline corrected, avg_ref all avg epochs')
    # plt.show()
    plt.savefig(f"/Users/simpleparadox/PycharmProjects/jwlab_eeg/eeg_visuals/cleaned_abs_remove_200uv/{participant}.png")

# participants = ["904", "905", "906", "908", "909", "910", "912", "913", "914", "916", "917", "919", "920", "921",
#                          "923", "924", "927", "928", "929", "930", "932"]
# participants = ["913", "914", "916", "917", "919", "920", "921",
#                          "923", "924", "927", "928", "929", "930", "932"]
participants = ["105", "106", "107", "109", "111", "112", "115", "116", "117", "119", "120", "121", "122", "124"]
participants = [121]
for participant in participants:
    try:
        print(participant)
        viz_eeg(participant)
    except Exception as e:
        print(e)


