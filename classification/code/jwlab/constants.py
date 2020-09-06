# ------------ WORD LIST ------------

word_list = ["baby", "BAD_STRING", "bird", "BAD_STRING", "cat", "dog", "duck", "mommy",
             "banana", "bottle", "cookie", "cracker", "BAD_STRING", "juice", "milk", "BAD_STRING"]


# ------------ OLD PARTICIPANTS ------------
#old_participants = ["107", "904", "905", "906"]
old_participants = []

# ------------ FILE PATH ------------
cleaned_data_filepath = ''
bad_trials_filepath = ''
db_filepath = ''
df_filepath = ''
df_filepath_sktime = ''

import sys
sys.path.insert(1, '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/classification/code')
from jwlab.profile import user
print(user)
if (user == "karl"):
    cleaned_data_filepath = "/home/kjslakov/projects/def-jwerker/kjslakov/data/cleaned/"
    bad_trials_filepath = "/home/kjslakov/projects/def-jwerker/kjslakov/data/datatracker/ML_badtrials-Table 1.csv"
    db_filepath = "/home/kjslakov/projects/def-jwerker/kjslakov/data/db/"
    df_filepath = "/home/kjslakov/projects/def-jwerker/kjslakov/data/ml_df_readys.pkl"
    df_filepath_sktime = "/home/kjslakov/projects/def-jwerker/kjslakov/data/ml_df_sktime.pkl"
elif (user == "jennlocal"):
    # ---- File path on network drive: ----
    db_filepath = "/Users/JennMacBook/Desktop/Studies/Animates EEG/7_Data/runningOffline/db/"
    cleaned_data_filepath = "/Users/JennMacBook/Desktop/Studies/Animates EEG/7_Data/runningOffline/cleaned/"
    bad_trials_filepath = "/Users/JennMacBook/Desktop/Studies/Animates EEG/7_Data/runningOffline/ML_badtrials-Table 1.csv"
    #db_filepath = "/Volumes/OFFCAMPUS/Jenn/Imported data/db/"
    #cleaned_data_filepath = "/Volumes/OFFCAMPUS/Jenn/Imported data/cleaned/"
    #bad_trials_filepath = "/Volumes/OFFCAMPUS/Jenn/Datatracker/ML_badtrials-Table 1.csv"
    #messy_trials_filepath = "/Volumes/OFFCAMPUS/Jenn/Datatracker/ML-Table 1.csv"
elif (user == "roxy"):
    # ---- File Path on Roxy local: ----
    db_filepath = "/Users/roxyk/Desktop/lab/db/"
    cleaned_data_filepath = "/Users/roxyk/Desktop/lab/cleaned/"
    bad_trials_filepath = "/Users/roxyk/Desktop/lab/Datatracker/ML_badtrials-Table 1.csv"
elif (user == "jenncc"):
    # ---- File Path on Jenn compute canada: ----
    db_filepath = "/home/campbejc/projects/def-campbejc/campbejc/data/lab/db/"
    cleaned_data_filepath = "/home/campbejc/projects/def-campbejc/campbejc/data/lab/cleaned/"
    bad_trials_filepath = "/home/campbejc/projects/def-campbejc/campbejc/data/lab/Datatracker/ML_badtrials-Table 1.csv"

elif (user == 'rohan'):
    print("user rohan")
    # ---- File path for Rohan on Local machine.
    cleaned_data_filepath = "Z:\\Jenn\\Data\\Imported data\\cleaned\\"
    bad_trials_filepath = "Z:\\Jenn\\Datatracker\\ML_badtrials-Table 1.csv"
    db_filepath = "Z:\\Jenn\\Data\\Imported data\\db\\"
    # df_filepath = "Z:\\Jenn\\ml_df_readys.pkl"
    # df_filepath_sktime = "Z:\\Jenn\\ml_df_sktime.pkl"

elif (user == 'rohancc'):
    # print("user rohan compute canada")
    # ---- File path for Rohan on Compute Canada.
    cleaned_data_filepath = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/cleaned/"
    bad_trials_filepath = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/ML_badtrials-Table 1.csv"
    db_filepath = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/db/"
    # df_filepath = "Z:\\Jenn\\ml_df_readys.pkl"
    # df_filepath_sktime = "Z:\\Jenn\\ml_df_sktime.pkl"