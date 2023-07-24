import math
import random
import numpy as np
import pandas as pd
import warnings

# Utilities

# Shuffle arrays together in sync
empty = np.zeros((0, 0))


def __shuffle(x, y=empty, z=empty):
    order = np.random.permutation(x.shape[0])
    x_shuffled = x[order, :]
    y_flag = False
    z_flag = False
    if y.shape[0] > 0:
        assert y.shape[0] == x.shape[0], "Arrays must have the same length."
        y_shuffled = y[order, :]
        y_flag = True

    if z.shape[0] > 0:
        assert z.shape[0] == x.shape[0], "Arrays must have the same length."
        z_shuffled = z[order, :]
        z_flag = True

    # Accomodate different number of outputs
    if y_flag and z_flag:
        return x_shuffled, y_shuffled, z_shuffled
    elif y_flag and not z_flag:
        return x_shuffled, y_shuffled
    elif not y_flag and not z_flag:
        return x_shuffled


# Min/max normalization of data
def __normalize_data(data):  # Assumes columns = features and rows = samples
    mean = np.mean(data, axis=0)
    normRange = np.max(data, axis=0) - np.min(data, axis=0)  # np.std(data, axis=0)

    norm = np.true_divide((data - mean), normRange)

    # to handle the situation where the the denominator equals zero after
    # normalization in some columns, convert the resulting NaNs to 0s
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

    return norm, mean, normRange


class ModifiedLogisticRegression:
    b = 0
    c_hat = 0
    feature_weights = 0
    mean = 0
    normRange = 0

    def __init__(self, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, s):
        n, m = X.shape
        X, s = __shuffle(X, s)
        X, mean, normRange = __normalize_data(X)

        # Add a column of ones
        X = np.concatenate((np.ones((n, 1)), X), axis=1)

        feature_weights = np.ones((1, m + 1))
        b = 1

        for i in range(self.epochs):
            # shuffle data for this epoch
            X, s = __shuffle(X, s)

            # Cycle through each datasample (need to vectorize!)
            for t in range(n):
                # Calculate partial derivative components
                e_w = np.exp(np.dot(-feature_weights, X[t, :].T))
                d1 = b * b + e_w
                d2 = 1 + d1

                if math.isinf(e_w):
                    dw = np.zeros_like(X[t, :])
                else:
                    dw = ((s[t] - 1) / d1 + 1 / d2) * X[t, :] * e_w

                db = ((1 - s[t]) / d1 - 1 / d2) * 2 * b

                feature_weights = feature_weights + self.learning_rate * dw
                b = b + self.learning_rate * db

        # Estimate c=p(s=1|y=1) using learned b value
        c_hat = np.divide(1, (1 + b * b))

        self.b = b
        self.c_hat = c_hat
        self.feature_weights = feature_weights
        self.mean = mean
        self.normRange = normRange

    def predict_proba(self, X):
        n, m = X.shape

        X = np.concatenate((np.ones((n, 1)), X), axis=1)

        mean = np.concatenate(([0], self.mean))
        normRange = np.concatenate(([1], self.normRange))

        normalizedSample = (X - mean) / normRange
        normalizedSample = np.nan_to_num(
            normalizedSample, nan=0.0, posinf=0.0, neginf=0.0
        )

        e_w = np.exp(np.dot(-normalizedSample, np.transpose(self.feature_weights)))
        e_w = np.nan_to_num(e_w, nan=0.0, posinf=0.0, neginf=0.0)

        s_hat = 1.0 / (1 + (self.b**2) + e_w)
        y_hat = s_hat / self.c_hat

        return y_hat

    def predict(self, X):
        predicted_proba = self.predict_proba(X)

        # Convert to list of bools and multiply
        preds = (predicted_proba >= 0.5) * 1
        return preds
