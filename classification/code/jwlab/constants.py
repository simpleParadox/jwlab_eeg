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
    cleaned_data_filepath = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/cleaned2/"
    bad_trials_filepath = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/ML_badtrials-Table 2.csv"
    db_filepath = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/db2/"
    # df_filepath = "Z:\\Jenn\\ml_df_readys.pkl"
    # df_filepath_sktime = "Z:\\Jenn\\ml_df_sktime.pkl"

    # 0.3Hz high-pass filtering.
    # cleaned_data_filepath = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/0point3Hzto50Hz/"

    # E65 reference.
    cleaned_data_filepath_e65 = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/band_pass_original_e65/"



    # Bad trials removed data and supporting files.
    cleaned_data_filepath_bad_remove = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/band_pass_original_bad_remove/"
    bad_remove_db_filepath = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/db_bad_remove_processed/"


    # no_detrending_low_pass_only_reref_with and without baseline.
    no_detrending_low_pass_only_reref_with_baseline_filepath = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/no_detrending_low_pass_only_reref_with_baseline/'
    no_detrending_low_pass_only_reref_no_baseline_filepath = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/no_detrending_low_pass_only_reref_no_baseline/'

    # ADAM detrending data low pass only reref with and without baseline.
    detrending_low_pass_only_reref_with_baseline_filepath = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/adam_detrending_low_pass_only_reref_with_baseline/csv/'
    detrending_low_pass_only_reref_no_baseline_filepath = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/adam_detrending_low_pass_only_reref_no_baseline/csv/'

    # Causal filtering.
    cleaned2_causal_with_baseline = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/band_pass_original_causal/'
    cleaned2_causal_no_baseline = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/band_pass_original_causal_no_base/'

    # Butterworth filtering.
    cleaned2_causal_butter_with_baseline_1hz = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/band_pass_original_causal_with_base_butter/'
    cleaned2_causal_butter_with_baseline_01hz = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/data/band_pass_original_causal_with_base_butter_0.1/'



    # ADAM detrended data
    # adam_40_order_filepath = "/home/rsaha/scratch/jwlab_eeg/data/adam_detrend_low_no_bad_40_order/"
    # adam_30_order_filepath = "/home/rsaha/scratch/jwlab_eeg/data/adam_detrend_csv_low_no_bad_remove/"