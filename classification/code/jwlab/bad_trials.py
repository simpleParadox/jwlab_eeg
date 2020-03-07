import numpy as np
import pandas as pd
from math import isnan
from jwlab.constants import bad_trials_filepath
from jwlab.constants import db_filepath
from jwlab.constants import messy_trials_filepath

def get_bad_trials(participants, ys, bad_trials_filepath):
    df = pd.read_csv(bad_trials_filepath)
    df.Ps = df.Ps.interpolate(method="pad")
    df = df[df['Reason'] != "left"]
    df = df.drop(columns=["Reason"], axis=1)

    ybad = []
    trial_count = []
    for i in range(len(participants)):
        p_df = df[df.Ps == int(participants[i])]
        bad_trials_count = p_df.shape[0]
        print("The number of bad trials of the participant - [%s] that are removed is [%d]." %(participants[i], bad_trials_count))
        if len(p_df) == 0:
            ybad.append([])
        elif isnan(p_df.tIndex.values[0]):
            ybad.append(get_ybad_from_cel_obs(participants, i, ys, df, p_df))
        else:
            ybad.append(p_df.tIndex.values.tolist())
        # append bad trials from the summary table
        messy_trials_df = pd.read_csv(messy_trials_filepath)
        messy_trials_df = messy_trials_df[messy_trials_df['PS'] == int(participants[i])]
        messy_trials_df = messy_trials_df[messy_trials_df['MessyData_Jenn'].notnull()]
        messy_string = messy_trials_df.MessyData_Jenn.values
        messy_list = []
        if len(messy_string) == 1:
            messy_list = messy_string[0].split(",")
        messy_list = [s for s in messy_list if s.isdigit()]
        bad_channels_count = len(messy_list)
        print("The number of bad channels that are removed is - [%d]." % len(messy_list))

        ybad[len(ybad)-1] = ybad[len(ybad)-1] + messy_list
        trial_count += [bad_channels_count + bad_trials_count]

    ybad = [[int(y) for y in x] for x in ybad]
    return ybad, trial_count

def get_ybad_from_cel_obs(participants, i, ys, df, p_df):
    ret = []
    db = pd.read_csv("%s%s_trial_cell_obs.csv" % (db_filepath, participants[i]))
    for index, row in df.iterrows():
        ret=np.append(ret,db[(db['cell'] == row['Cell']) & (db['obs'] == row['Observation'])
                             & (int(participants[i]) == row['Ps'])].trial_index.values)
    return ret.tolist()

def transform_ybad_indices(ybad, ys):
    offset = 0
    for i in range(len(ybad)):
        ybad[i] = np.array(ybad[i]) + offset
        offset += len(ys[i])
    return np.concatenate(ybad).astype(np.int32)

def get_left_trials_each_word(participants):
    # get the list that contains the count for each word (cell) originally

    # read the df
    for p in range(len(participants)):
        #print(p)
        orig_df = pd.read_csv("%s%s_trial_cell_obs.csv" % (db_filepath, participants[p]))
        p_df = orig_df.groupby(['cell']).count()
        bad_trials_df = pd.read_csv(bad_trials_filepath)
        bad_trials_df.Ps = bad_trials_df.Ps.interpolate(method="pad")
        bad_trials_df = bad_trials_df[bad_trials_df['Reason'] != "left"]
        bad_trials_df = bad_trials_df[bad_trials_df.Ps == int(participants[p])]
        bad_trials_df = bad_trials_df.groupby(['Cell']).count()
        bad_trials_df = bad_trials_df.drop(columns=['Ps', 'Reason', 'tIndex'])

        messy_trials_df = pd.read_csv(messy_trials_filepath)
        messy_trials_df = messy_trials_df[messy_trials_df['PS'] == int(participants[p])]
        messy_trials_df = messy_trials_df[messy_trials_df['MessyData_Jenn'].notnull()]
        messy_string = messy_trials_df.MessyData_Jenn.values
        messy_list = []
        if len(messy_string) == 1:
            messy_list = messy_string[0].split(",")
        messy_list = [s for s in messy_list if s.isdigit()]
        messy_list = [int(s) for s in messy_list]
        if messy_list:
            for i in messy_list:
                messy_df = orig_df.drop(i-1)

            messy_df = messy_df.groupby(['cell']).count()
            messy_df = messy_df.drop(columns=['trial_index'])

            messy_df = messy_df.rename(columns={"obs": "Observation"})

            df_add = messy_df.subtract(bad_trials_df, fill_value=0)
            #print(df_add)
        else:
            df_add = p_df.subtract(bad_trials_df, fill_value=0)

    # get the list that contains the count for each word that will be removed
        # from bad trial csv
        # from messy data
