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
    for i in range(len(participants)):
        p_df = df[df.Ps == int(participants[i])]
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
        ybad[len(ybad)-1] = ybad[len(ybad)-1] + messy_list
    ybad = [[int(y) for y in x] for x in ybad]
    return ybad

def get_ybad_from_cel_obs(participants, i, ys, df, p_df):
    ret = []
    db = pd.read_csv("%s%s_trial_cell_obs.csv" % (db_filepath, participants[i]))
    for row in df.iterrows():
        ret=np.append(ret,db[(db['cell'] == row[1]['Cell']) & (db['obs'] == row[1]['Observation']) 
                             & (int(participants[i]) == row[1]['Ps'])].trial_index.values)
    return ret.tolist()

def transform_ybad_indices(ybad, ys):
    offset = 0
    for i in range(len(ybad)):
        ybad[i] = np.array(ybad[i]) + offset
        offset += len(ys[i])     
    return np.concatenate(ybad).astype(np.int32)
