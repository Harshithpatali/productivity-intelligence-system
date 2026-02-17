import numpy as np
import pandas as pd


def oversample_minority(df, target_col="target"):

    classes = df[target_col].unique()

    class_counts = df[target_col].value_counts()
    max_count = class_counts.max()

    balanced_df = []

    for c in classes:

        class_df = df[df[target_col] == c]

        if len(class_df) < max_count:

            extra = class_df.sample(
                max_count - len(class_df),
                replace=True,
                random_state=42,
            )

            class_df = pd.concat([class_df, extra])

        balanced_df.append(class_df)

    balanced_df = pd.concat(balanced_df)

    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
