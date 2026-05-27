import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def rms_error(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def safe_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])