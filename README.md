# NeuroMatrix LSM

NeuroMatrix is a lightweight supervised matrix-learning model for adaptive prediction based on statistical transformations.

## Install
```bash
pip install neuromatrix-lsm
```

## Usage
```python
from neuromatrix import NeuroMatrixRegressor
import numpy as np

X = np.array(, dtype=float)
y = np.array(, dtype=float)[4][5][1]

model = NeuroMatrixRegressor()
model.fit(X, y)
pred = model.predict()
print(pred)
```

## License
MIT