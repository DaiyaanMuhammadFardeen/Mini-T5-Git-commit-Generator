import pandas as pd
import gc

# Load the full dataset
df = pd.read_parquet("commitpack_cleaned.parquet")
total_len = len(df)
one_third = total_len // 3

# ---------------- Part 1 ----------------
df_part1 = df.iloc[:one_third].reset_index(drop=True)
df_part1.to_parquet("commitpack_part1.parquet")
del df_part1
gc.collect()

# ---------------- Part 2 ----------------
df_part2 = df.iloc[one_third:2*one_third].reset_index(drop=True)
df_part2.to_parquet("commitpack_part2.parquet")
del df_part2
gc.collect()

# ---------------- Part 3 ----------------
df_part3 = df.iloc[2*one_third:].reset_index(drop=True)
df_part3.to_parquet("commitpack_part3.parquet")
del df_part3
gc.collect()

# Free main dataframe
del df
gc.collect()

