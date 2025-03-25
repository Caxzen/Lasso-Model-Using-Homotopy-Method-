# Lasso Homotopy Model

This repository contains an implementation of a Lasso Homotopy algorithm, a method for solving the Lasso (Least Absolute Shrinkage and Selection Operator) regression problem by tracing the regularization path from a high penalty to a user-specified target penalty. The model is built from scratch using NumPy and follows the homotopy continuation approach described in literature like "Pathwise Coordinate Optimization" by Friedman et al.

## 1. What Does the Model Do and When Should It Be Used?

### 1.1 What It Does
The `LassoHomotopyModel` performs linear regression with L1 regularization, producing sparse solutions where many coefficients are exactly zero. It starts with a high regularization parameter (`lambda_max`), where all coefficients are zero, and iteratively reduces the penalty along the regularization path until reaching the user-defined `lambda_reg`. This process:
- Identifies the most influential features by activating them into the model as the penalty decreases.
- Balances model fit (minimizing squared error) and sparsity (penalizing the L1 norm of coefficients).

### 1.2 When to Use It
Use this model when:
- **Feature Selection**: You need to identify a subset of important predictors in a high-dimensional dataset (e.g., when features outnumber samples).
- **Collinearity**: Your data has highly correlated features, and you want a sparse solution to avoid overfitting (e.g., `collinear_data.csv`).
- **Interpretability**: You prefer a model where coefficients directly indicate feature importance, with many set to zero.
- **Research or Education**: You’re studying or teaching the mechanics of Lasso regression and its homotopy path.

It’s particularly useful compared to standard solvers when you want to explore the entire regularization path or customize the homotopy process, though it may not yet match the optimization efficiency of libraries like `scikit-learn` for large-scale production use.

## 2. How Did You Test Your Model to Determine If It Is Working Reasonably Correctly?

The model’s correctness and robustness are validated through a comprehensive test suite in `tests/test_LassoHomotopy.py`, using `pytest`. Key tests include:

1. **Small Test Dataset (`small_test.csv`)**:
   - Fits the model across multiple `lambda_reg` values (0.01, 0.1, 1.0, 10.0).
   - Asserts MSE < 50, targeting a low error (though prior runs showed ~530, indicating tuning needed).
   - Prints MSE, predictions, and sparsity for debugging.

2. **Collinear Data (`collinear_data.csv`)**:
   - Ensures sparse solutions (non-zero coefficients ≤ half the features) due to collinearity.
   - Asserts MSE < 30, expecting tight fit (prior single-sample MSE ~0.0003).
   - Checks for non-zero predictions and collinearity detection.

3. **Noise Robustness (`test_mse_with_noise`)**:
   - Adds Gaussian noise (std=0.1) to `small_test.csv`’s target.
   - Compares MSEs (original vs. noisy), asserting both < 50 and difference < 50.
   - Verifies stability in coefficient sparsity.

4. **Sklearn Comparison (`test_collinear_data_sklearn_comparison`)**:
   - Fits both our model and `sklearn.linear_model.Lasso` on `collinear_data.csv`.
   - Asserts MSEs < 30 (ours) and < 10 (sklearn), with difference < 25.
   - Confirms sparsity and coefficient consistency with a trusted baseline.

5. **Singular Matrix (`test_singular_matrix`)**:
   - Tests a perfectly collinear dataset (e.g., `X_1 = 2 * X_0`).
   - Asserts MSE < 20, ensuring numerical stability with a small regularization term (`1e-6`).

These tests cover correctness (vs. sklearn), sparsity, noise robustness, and edge cases. However, the high MSE (~530) on `small_test.csv` suggests the model needs refinement to hit lower targets (e.g., 85), as discussed below.

## Visualizations
See `LassoHomotopyDemo.ipynb` for interactive visualizations:
- **Coefficient Paths**: Tracks feature activation over `lambda`.
- **Prediction Plots**: Compares true vs. predicted values.
- **Sklearn Comparison**: MSE and sparsity vs. `sklearn.linear_model.Lasso`.


## 3. What Parameters Have You Exposed to Users to Tune Performance?

The model exposes one key parameter for tuning:

- **`lambda_reg` (default: 0.1)**:
  - **Description**: The target regularization strength in the Lasso objective: `1/2 ||y - Xθ||_2^2 + lambda_reg * ||θ||_1`.
  - **Tuning**: 
    - Smaller values (e.g., 0.01) reduce sparsity, allowing more non-zero coefficients and potentially lowering MSE but risking overfitting.
    - Larger values (e.g., 10.0) increase sparsity, setting more coefficients to zero, useful for feature selection or noisy data.
  - **Usage**: Set via `LassoHomotopyModel(lambda_reg=0.01)` to adjust the trade-off between fit and sparsity.

Other internal parameters (e.g., `max_iterations=100`, `gamma` cap at 1.0) are fixed but could be exposed with further development for finer control.

## 4. Are There Specific Inputs That Your Implementation Has Trouble With? Could You Work Around These Given More Time?

### Trouble Spots
1. **Sligth MSE Difference than scikit on `small_test.csv`**:
   - Current MSE (~20) exceeds the target by 10.
   
2. **Large Datasets**:
   - Untested on >5000 samples or >30 features; runtime may scale poorly.
   - **Cause**: Due to Iterative matrix inversions in each step.
   - **Workaround**: Optimize with sparse matrices or batch processing.

### Time and Feasibility
- **Given More Time**: All issues are addressable:
  - **MSE Tuning**: Adjust `gamma` logic (e.g., adaptive steps) and validate against sklearn’s.
  - **Scalability**: Profile and optimize matrix operations (e.g., Numba or sparse support).
- **Fundamental Limits**: current gaps are implementation details, not theoretical flaws. 

## Usage
```python
from model.LassoHomotopy import LassoHomotopyModel
import numpy as np

X = np.loadtxt("tests/small_test.csv", delimiter=",", skiprows=1)[:, :-1]
y = np.loadtxt("tests/small_test.csv", delimiter=",", skiprows=1)[:, -1]
model = LassoHomotopyModel(lambda_reg=0.1)
results = model.fit(X, y)
preds = results.predict(X)
print(f"MSE: {np.mean((y - preds) ** 2)}")
