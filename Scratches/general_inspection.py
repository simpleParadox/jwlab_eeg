import pandas as pd
import numpy as np


participants = ["904", "905", "906", "908", "909", "910", "912", "913", "914", "916", "917", "919", "920", "921",
                         "923", "924", "927", "928", "929", "930", "932"]

temp_df_len = 0
for participant in participants:
    path = f"/Users/simpleparadox/Desktop/Projects/jwlab_eeg/Data/Imported/adam_detrend_csv_low_no_bad_remove/{participant}_cleaned_ml.csv"
    data = pd.read_csv(path)

    temp_df_len += len(data)
    if temp_df_len % 100 !=0:
        print(participant)
        break