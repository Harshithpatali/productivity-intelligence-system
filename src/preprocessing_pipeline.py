import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data_cleaning import clean_data
from src.feature_engineering import create_features
from src.config import *
import joblib




def run_preprocessing():

    df = pd.read_csv(DATA_PATH)

    # Cleaning
    df = clean_data(df)

    # Feature engineering
    df = create_features(df)

    # Encode target
    encoder = LabelEncoder()
    df["productivity_level"] = encoder.fit_transform(df["productivity_level"])

    # Save encoder
    joblib.dump(encoder, ENCODER_PATH)

    # Features / target split
    X = df.drop(["productivity_level"], axis=1)
    y = df["productivity_level"]
    # Save feature column names
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df["target"] = y

    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Preprocessing completed!")


if __name__ == "__main__":
    run_preprocessing()
