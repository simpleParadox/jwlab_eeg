import numpy as np
import pandas as pd
from math import isnan
from constants import bad_trials_filepath

df = pd.read_csv(bad_trials_filepath)
df = df.drop(columns=["Reason"], axis=1)
df.Ps = df.Ps.interpolate(method="pad")

def get_bad_trials(participants, ys):
    ybad = []
    for i in range(len(participants)):
        p_df = df[df.Ps == int(participants[i])]
        if len(p_df) == 0:
            ybad.append([])
        elif isnan(p_df.tIndex.values[0]):
            ybad.append(get_ybad_from_cel_obs(participants, i, ys, p_df))
        else:
            ybad.append(p_df.tIndex.values.tolist())
    return ybad

def get_ybad_from_cel_obs(participants, i, ys, p_df):
    cumsums = []
    for j in range(1, int(p_df.Cell.max() + 1)):
        # from https://stackoverflow.com/questions/38949308/find-the-nth-time-a-specific-value-is-exceeded-in-numpy-array
        cond = np.array(ys[i]) == j
        cumsums.append(np.cumsum(cond))
    return [np.searchsorted(cumsums[int(df.iloc[k].Cell) - 1], int(df.iloc[k].Observation)) for k in p_df.index]

def transform_ybad_indices(ybad, ys):
    offset = 0
    for i in range(len(ybad)):
        ybad[i] = np.array(ybad[i]) + offset
        offset += len(ys[i])
        
    return np.concatenate(ybad).astype(np.int32)
