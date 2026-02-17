import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype != "object":
            df[col].fillna(df[col].median(), inplace=True)

    # Outlier clipping
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower, upper)

    return df
