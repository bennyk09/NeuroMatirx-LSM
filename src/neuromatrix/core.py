import numpy as np
from .utils import sigmoid, rms_error, safe_corr

class NeuroMatrixRegressor:
    def __init__(self, alpha=0.01, k_init=0.5, max_iter=500, tol=1e-6):
        self.alpha = alpha
        self.k_init = k_init
        self.max_iter = max_iter
        self.tol = tol
        self.k_ = None
        self.r_ = None
        self.rms_ = None
        self.history_ = []
        self.is_fitted_ = False
        self._x_train_s = None
        self._y_train_s = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        Xs = sigmoid(X)
        ys = sigmoid(y)

        k = float(self.k_init)
        prev_rms = None

        for epoch in range(self.max_iter):
            r = safe_corr(Xs, ys)
            lam = Xs + k * r * (ys - Xs)
            y_pred = sigmoid(lam)
            rms = rms_error(y_pred, ys)

            self.history_.append({
                "epoch": epoch + 1,
                "k": k,
                "r": r,
                "rms": rms
            })

            if prev_rms is not None and abs(prev_rms - rms) < self.tol:
                break

            k = k - self.alpha * rms
            prev_rms = rms

        self.k_ = k
        self.r_ = r
        self.rms_ = rms
        self._x_train_s = Xs
        self._y_train_s = ys
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict()")

        X = np.asarray(X, dtype=float).reshape(-1)
        Xs = sigmoid(X)
        lam = Xs + self.k_ * self.r_ * (self._y_train_s.mean() - Xs)
        y_pred = sigmoid(lam)
        return y_pred

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)