## Function to find unique values in each column
import pandas as pd

def unique_values_per_categorical_column(df):
    unique_count = {}
    unique_values = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count[col] = df[col].nunique()
            unique_values[col] = df[col].unique().tolist()

    return unique_count, unique_values