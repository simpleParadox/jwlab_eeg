import pandas as pd

FILEPATH = "../../../data/ml_ready/999_cleaned_ml.csv"

df = pd.read_csv(FILEPATH)
print(df['Time'])