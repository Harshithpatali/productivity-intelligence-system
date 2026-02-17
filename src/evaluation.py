import numpy as np
import pandas as pd
import joblib

from src.config import *

from src.model_training import NeuralNetwork

# ======================
# Confusion Matrix
# ======================

def confusion_matrix(y_true, y_pred, num_classes):

    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    return matrix


# ======================
# Precision / Recall / F1
# ======================

def classification_metrics(cm):

    num_classes = cm.shape[0]

    precision = []
    recall = []
    f1 = []

    for i in range(num_classes):

        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP

        p = TP / (TP + FP + 1e-8)
        r = TP / (TP + FN + 1e-8)

        precision.append(p)
        recall.append(r)

        f1.append(2 * p * r / (p + r + 1e-8))

    return precision, recall, f1


# ======================
# Accuracy
# ======================

def accuracy_score(y_true, y_pred):

    return np.mean(y_true == y_pred)


# ======================
# Evaluation Pipeline
# ======================

def evaluate_model():

    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop("target", axis=1).values
    y = df["target"].values.astype(int)

    model = joblib.load(MODEL_PATH)

    preds = model.predict(X)

    acc = accuracy_score(y, preds)

    num_classes = len(np.unique(y))
    cm = confusion_matrix(y, preds, num_classes)

    precision, recall, f1 = classification_metrics(cm)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", cm)

    for i in range(num_classes):
        print(
            f"\nClass {i} → Precision: {precision[i]:.3f}, "
            f"Recall: {recall[i]:.3f}, F1: {f1[i]:.3f}"
        )


if __name__ == "__main__":
    evaluate_model()
