# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE_MAGIC_CELL
# Automatically replaced inline charts by "no-op" charts
# %pylab inline
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read the dataset as a Pandas dataframe in memory
# Note: here, we only read the first 100K rows. Other sampling options are available
dataset_customers_web_joined = dataiku.Dataset("customers_web_joined")
df = dataset_customers_web_joined.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from python_library.myfunctions import bin_values

# Apply the function to create the high_value column then drop the revenue column
revenue_cutoff = int(dataiku.get_custom_variables()["revenue_cutoff"])
df['high_value'] = df.revenue.apply(bin_values, v=revenue_cutoff)
df.drop(columns=['revenue'], inplace=True)
# df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
customers_binned = dataiku.Dataset("customers_binned")
customers_binned.write_with_schema(df)