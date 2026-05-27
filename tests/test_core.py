import numpy as np
from neuromatrix import NeuroMatrixRegressor

def test_fit_predict_runs():
    X = np.array([800, 1000, 1200, 1500, 1800], dtype=float)
    y = np.array([40, 50, 60, 75, 90], dtype=float)

    model = NeuroMatrixRegressor(max_iter=50)
    preds = model.fit_predict(X, y)

    assert len(preds) == len(y)
    assert model.is_fitted_