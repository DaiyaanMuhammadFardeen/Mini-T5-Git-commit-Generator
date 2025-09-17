import dask.dataframe as dd
import shutil
import os
from dask.diagnostics import ProgressBar

# Read all parquet files
df = dd.read_parquet("parquets/*.parquet")

# Temporary output folder
temp_dir = "combined_temp.parquet"
output_file = "combined.parquet"

# Enable progress bar
with ProgressBar():
    df.repartition(npartitions=1).to_parquet(
        temp_dir,
        engine="pyarrow",
        write_index=False
    )

# Move the single parquet file out of the folder
for f in os.listdir(temp_dir):
    if f.endswith(".parquet"):
        shutil.move(os.path.join(temp_dir, f), output_file)
        break

# Clean up temp directory
shutil.rmtree(temp_dir)

print(f"✅ Combined parquet saved as {output_file}")

