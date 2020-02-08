import numpy as np
import pandas as pd
from math import isnan
from constants import bad_trials_filepath

df = pd.read_csv(bad_trials_filepath)
df.Ps = df.Ps.interpolate(method="pad") #filling up all the blank rows at column = Ps in the sheet, by repeating the most recently seen values  
#df = df[df.Reason != 'left']
df = df.drop(columns=["Reason"], axis=1)

def get_bad_trials(participants, ys):
    ybad = []
   
    for i in range(len(participants)):
        p_df = df[df.Ps == int(participants[i])] #getting the dataframe for the specific participant
        if len(p_df) == 0:
            ybad.append([])
        elif isnan(p_df.tIndex.values[0]): #if the rows at the column=tIndex is nan (not filled in by numbers), do ..
            ybad.append(get_ybad_from_cel_obs(participants, i, ys, p_df))
        else:
            ybad.append(p_df.tIndex.values.tolist())
            #print(ybad)
            # otherwise just append the indices from column=tIndex defined in the sheet
    
    return ybad

def get_ybad_from_cel_obs(participants, i, ys, p_df):
    cumsums = []
    for j in range(1, int(p_df.Cell.max() + 1)): # iterating based on the number filled in the column=Cell
        # from https://stackoverflow.com/questions/38949308/find-the-nth-time-a-specific-value-is-exceeded-in-numpy-array
        cond = np.array(ys[i]) == j #generating an array with booleans, true entries for the Cell_in_ys = current Cell
        cumsums.append(np.cumsum(cond)) #an array of cumulative sums of booleans (true=1, false=0)
    # p_df.index is the index of the corresponding participant's rows in the sheet
    # df.iloc[k] gives the row of the corresponding row number
    # searchsorted insert the second element to a sorted array (the first element), returns the index for insertion, which gives 
    # the indices for the bad_trails in the original dataset         
    return [np.searchsorted(cumsums[int(df.iloc[k].Cell) - 1], int(df.iloc[k].Observation)) for k in p_df.index]

def transform_ybad_indices(ybad, ys):
    offset = 0
    for i in range(len(ybad)):
        ybad[i] = np.array(ybad[i]) + offset
        offset += len(ys[i])
        print(ybad[i])
        
    return np.concatenate(ybad).astype(np.int32)
