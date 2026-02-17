import numpy as np
import pandas as pd
import joblib

from src.config import *
from src.sampling import oversample_minority


class NeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        lr=0.005,
        lambda_l2=0.001,
        dropout_rate=0.2,
        class_weights=None,
    ):

        self.lr = lr
        self.lambda_l2 = lambda_l2
        self.dropout_rate = dropout_rate
        self.class_weights = class_weights

        # He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    # ---------- ACTIVATIONS ----------

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    # ---------- DROPOUT ----------

    def apply_dropout(self, A):
        mask = (np.random.rand(*A.shape) > self.dropout_rate) / (1 - self.dropout_rate)
        return A * mask, mask

    # ---------- FORWARD ----------

    def forward(self, X, training=True):

        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        if training:
            self.A1, self.dropout_mask = self.apply_dropout(self.A1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)

        return self.A2

    # ---------- LOSS ----------

    def compute_loss(self, y_true, y_pred):

        m = y_true.shape[0]

        weights = self.class_weights[y_true] if self.class_weights is not None else 1

        log_likelihood = -np.log(y_pred[range(m), y_true]) * weights
        data_loss = np.sum(log_likelihood) / m

        l2_loss = (
            self.lambda_l2 *
            (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
            / (2 * m)
        )

        return data_loss + l2_loss

    # ---------- BACKWARD ----------

    def backward(self, X, y_true):

        m = X.shape[0]

        y_pred = self.A2
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[np.arange(m), y_true] = 1

        weights = self.class_weights[y_true][:, None] if self.class_weights is not None else 1

        dZ2 = (y_pred - y_one_hot) * weights
        dW2 = np.dot(self.A1.T, dZ2) / m + self.lambda_l2 * self.W2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dA1 *= self.dropout_mask

        dZ1 = dA1 * self.relu_derivative(self.Z1)

        dW1 = np.dot(X.T, dZ1) / m + self.lambda_l2 * self.W1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ---------- MINI BATCH TRAIN ----------

    def train(self, X, y, epochs=2000, batch_size=64, patience=200):

        best_loss = float("inf")
        patience_counter = 0

        n = X.shape[0]

        for epoch in range(epochs):

            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n, batch_size):

                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                y_pred = self.forward(X_batch, training=True)
                loss = self.compute_loss(y_batch, y_pred)

                self.backward(X_batch, y_batch)

            # Full loss for monitoring
            full_pred = self.forward(X, training=False)
            full_loss = self.compute_loss(y, full_pred)

            if epoch % 50 == 0:
                acc = self.accuracy(X, y)
                print(f"Epoch {epoch}, Loss: {full_loss:.4f}, Acc: {acc:.4f}")

            # Early stopping
            if full_loss < best_loss:
                best_loss = full_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > patience:
                print("Early stopping triggered")
                break

    def predict(self, X):
        probs = self.forward(X, training=False)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


# ======================
# TRAIN PIPELINE
# ======================

def compute_class_weights(y):

    classes = np.unique(y)
    weights = {}

    total = len(y)

    for c in classes:
        count = np.sum(y == c)
        weights[c] = total / (len(classes) * count)

    return weights


def train_model():

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Balance data
    df = oversample_minority(df)

    X = df.drop("target", axis=1).values
    y = df["target"].values.astype(int)

    class_weights_dict = compute_class_weights(y)
    class_weights = np.array([class_weights_dict[i] for i in range(len(class_weights_dict))])

    model = NeuralNetwork(
        input_size=X.shape[1],
        hidden_size=64,
        output_size=len(np.unique(y)),
        lr=0.005,
        lambda_l2=0.001,
        dropout_rate=0.2,
        class_weights=class_weights,
    )

    model.train(X, y)

    joblib.dump(model, MODEL_PATH)

    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()
