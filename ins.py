# stage1_load_inspect.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# --- 1. load dataset (cleaned 'data' config) ---
# This downloads the dataset from Hugging Face the first time.
dset = load_dataset("Quoron/EEG-semantic-text-relevance", "data")

# Inspect splits and columns
print(dset)                          # shows available splits and sizes
print("Columns:", dset["train"].column_names)

# Peek a few rows as a Pandas DataFrame
df_head = dset["train"].to_pandas().head(5)
print(df_head.T)                     # transposed for readability
