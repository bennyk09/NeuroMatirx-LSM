# NeuroMatirx-LSM

NeuroMatirx-LSM is a lightweight Python package that provides a simple interface for training and using the NeuroMatrix model for supervised prediction tasks. The package is designed as a research-oriented implementation that turns the mathematical workflow from the project into a reusable Python library.

## Overview

This project packages the NeuroMatrix method into an installable Python module so it can be imported and used in scripts, notebooks, and experiments. The package is intended to make the workflow easier to test, document, and publish.

The core goals of the project are:

- Provide a clean Python API for model training and prediction.
- Convert the NeuroMatrix research idea into reusable code.
- Support experimentation with simple numerical datasets.
- Make the project easier to distribute through GitHub and PyPI.

## Project Structure

A typical project structure for this package is shown below:

```text
NeuroMatirx-LSM/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── neuromatrix/
│       ├── __init__.py
│       ├── core.py
│       ├── utils.py
│       └── exceptions.py
├── tests/
│   └── test_core.py
└── examples/
    └── main.py
```

### Folder Description

- `pyproject.toml` — package metadata, dependencies, and build configuration.
- `README.md` — project documentation and usage guide.
- `LICENSE` — license information for the package.
- `src/neuromatrix/` — main package source code.
- `tests/` — unit tests for checking correctness.
- `examples/` — example scripts showing how to use the package.

## Installation

### Install locally for development

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/bennyk09/NeuroMatirx-LSM.git
cd NeuroMatirx-LSM
pip install -e .
```

### Install dependencies manually

If needed, install the required packages first:

```bash
pip install numpy
```

## Usage

Below is a simple example of how to train and use the model.

```python
from neuromatrix import NeuroMatrixRegressor
import numpy as np

X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
], dtype=float)

y = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)

model = NeuroMatrixRegressor()
model.fit(X, y)
pred = model.predict(X)

print("Predictions:", pred)
```

## How It Works

The package is built around the idea of turning the NeuroMatrix mathematical procedure into code that can be executed through a class-based interface. At a high level, the workflow is:

1. Accept input data `X` and target data `y`.
2. Normalize the values using the package's internal preprocessing logic.
3. Apply the NeuroMatrix transformation process.
4. Update internal parameters during fitting.
5. Generate predictions for new or existing inputs.
6. Return model outputs and evaluation-related values.

This makes the project easier to test and reuse than keeping the method only in standalone scripts.

## Main Features

- Clean object-oriented API.
- Fit and predict workflow.
- Reusable package structure.
- Easy integration with NumPy-based datasets.
- Suitable for research prototypes and academic experimentation.

## API Design

The main object exposed by the package is expected to be:

```python
NeuroMatrixRegressor
```

Typical methods:

- `fit(X, y)` — train the model using input and target data.
- `predict(X)` — generate predictions for input data.
- `fit_predict(X, y)` — optional helper method for quick usage.

Possible attributes after training:

- `k_` — learned or updated internal parameter.
- `rms_` — root mean square error value.
- `is_fitted_` — indicates whether the model has been trained.

## Example Script

A simple `main.py` example:

```python
from neuromatrix import NeuroMatrixRegressor
import numpy as np

X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
], dtype=float)

y = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)

model = NeuroMatrixRegressor()
model.fit(X, y)
predictions = model.predict(X)

print(predictions)
```

To run the script on Windows PowerShell:

```powershell
python main.py
```

Do not type only `main.py`, because PowerShell does not execute files from the current folder by default.

## Development Workflow

A typical development workflow for this project is:

1. Implement package code in `src/neuromatrix/`.
2. Write or update tests in `tests/`.
3. Run example scripts from `examples/` or the project root.
4. Build the package:

```bash
python -m build
```

5. Install locally for testing:

```bash
pip install -e .
```

## Publishing

To publish the package using GitHub Actions and PyPI trusted publishing:

1. Make sure `pyproject.toml` contains the correct project name and version.
2. Commit and push the project to GitHub.
3. Ensure `.github/workflows/python-publish.yml` exists.
4. Create a GitHub release such as `v0.1.0`.
5. Let GitHub Actions build and upload the package to PyPI.

## Common Errors

### PowerShell says `main.py` is not recognized

Use:

```powershell
python main.py
```

instead of:

```powershell
main.py
```

### Import error for `neuromatrix`

Make sure the package is installed first:

```bash
pip install -e .
```

### Invalid NumPy array syntax

This is wrong:

```python
X = np.array(, dtype=float)
```

This is correct:

```python
X = np.array([[1.0], [2.0], [3.0]], dtype=float)
```

## Future Improvements

Potential future improvements for the project include:

- Better validation and error handling.
- Support for more dataset shapes.
- Additional evaluation metrics.
- Better examples and notebooks.
- PyPI-ready release polishing.
- Expanded documentation for the underlying mathematical model.

## Contributing

Contributions can include code improvements, bug fixes, documentation updates, tests, and packaging enhancements.

Suggested contribution flow:

1. Fork the repository.
2. Create a new branch.
3. Make changes.
4. Test the package.
5. Submit a pull request.

## License

Add your preferred license in the `LICENSE` file.

## Author

Project repository:

[https://github.com/bennyk09/NeuroMatirx-LSM](https://github.com/bennyk09/NeuroMatirx-LSM)
