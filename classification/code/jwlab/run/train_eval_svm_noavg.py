import run_setup

import pandas as pd
import numpy as np

from jwlab.run.computecanada_constants import df_path
from jwlab.ml_prep import load_ml_df, y_to_binary, no_average, average_trials
from sklearn.svm import LinearSVC
from jwlab.eval import eval_normal

print("start")

df = load_ml_df(df_path)
print("loaded_df")

X,y,w,p = no_average(df)
print("prep train_data")

model = LinearSVC(C=1.0, max_iter=1000)
print("model")

errs = eval_normal(model, X, y, 1, random_state=0)

print(errs)
print(np.mean(errs))