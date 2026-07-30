import pandas as pd

print("Pandas version:", pd.__version__)

df = pd.read_csv(
    "mock_kuching_30days_heavy.csv"
)

print(df.head())
print(df.shape)

print("DONE")