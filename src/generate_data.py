import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 5000

# Generate features
sleep_hours = np.random.normal(7, 1.5, n_samples).clip(3, 10)
work_hours = np.random.normal(8, 2, n_samples).clip(2, 14)
screen_time = np.random.normal(6, 2, n_samples).clip(1, 12)
breaks = np.random.poisson(4, n_samples)
task_switches = np.random.poisson(10, n_samples)
stress_level = np.random.uniform(1, 10, n_samples)
hydration = np.random.uniform(1, 5, n_samples)
noise_level = np.random.uniform(1, 10, n_samples)

# Productivity logic
productivity_score = (
    sleep_hours * 10
    + work_hours * 5
    + hydration * 6
    - stress_level * 7
    - task_switches * 2
    - noise_level * 3
    + breaks * 2
)

# Normalize score
productivity_score = (productivity_score - productivity_score.min()) / (
    productivity_score.max() - productivity_score.min()
) * 100

# Convert to classes
def classify(score):
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"


productivity_level = [classify(s) for s in productivity_score]

# Fatigue logic
fatigue_score = (
    work_hours * 6
    + stress_level * 8
    - sleep_hours * 7
    - breaks * 3
)

fatigue_score = (fatigue_score - fatigue_score.min()) / (
    fatigue_score.max() - fatigue_score.min()
) * 100

data = pd.DataFrame({
    "sleep_hours": sleep_hours,
    "work_hours": work_hours,
    "screen_time": screen_time,
    "breaks": breaks,
    "task_switches": task_switches,
    "stress_level": stress_level,
    "hydration": hydration,
    "noise_level": noise_level,
    "productivity_score": productivity_score,
    "fatigue_score": fatigue_score,
    "productivity_level": productivity_level
})

data.to_csv("data/raw_productivity.csv", index=False)

print("Dataset created successfully!")
