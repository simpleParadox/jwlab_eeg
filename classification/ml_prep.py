import pandas as pd
import numpy as np

def prep_ml(filepath, participants):
    dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (filepath, s)) for s in participants]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    df = df[df.Time >= 0]
    df = df.drop(columns=["Time", "E65"], axis=1)

    X = df.values
    (i,j) = X.shape
    X = np.reshape(X, (i // 1000, j * 1000))

    ys = [np.loadtxt("%s%s_labels.txt" % (filepath, s)) for s in participants]
    y = np.concatenate(ys)
    return X, y
    