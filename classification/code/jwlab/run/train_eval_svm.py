import run_setup

import pandas as pd
import numpy as np

from jwlab.run.computecanada_constants import data_path
from jwlab.ml_prep import prep_ml_internal, load_ml_data
from sklearn.svm import LinearSVC

participants = ["107", "109", "111", "112", "115", "116", "904", "905", "906", "908", "909", "910", "912"]

unmodified_df, unmodified_ys = load_ml_data(data_path, participants)
unmodified_df2, unmodified_ys2 = unmodified_df.copy(), unmodified_ys2.copy()

X,y,p,w,df = prep_ml_internal(unmodified_df, unmodified_ys, participants, downsample_num=1000, averaging="no_averaging")
Xp,yp,pp,wp,df2 = prep_ml_internal(unmodified_df2, unmodified_ys2, participants, downsample_num=1000, averaging="average_trials")

model = LinearSVC(C=1.0, max_iter=1e9)


model.fit(X, y)
print("error: ", np.mean(model.predict(Xp) != yp))