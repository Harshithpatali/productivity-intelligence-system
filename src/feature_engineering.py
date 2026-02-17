import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df["work_sleep_ratio"] = df["work_hours"] / (df["sleep_hours"] + 1)
    df["stress_per_hour"] = df["stress_level"] / (df["work_hours"] + 1)
    df["break_efficiency"] = df["breaks"] / (df["work_hours"] + 1)
    df["focus_index"] = df["hydration"] / (df["task_switches"] + 1)
    df["screen_fatigue"] = df["screen_time"] * df["stress_level"]

    return df
