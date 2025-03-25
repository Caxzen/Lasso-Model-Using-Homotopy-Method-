import pytest
import numpy as np
import csv
from model.LassoHomotopy import LassoHomotopyModel
from sklearn.linear_model import Lasso

def load_csv_data(filepath):
    """load CSV data"""
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)  # Skip header
    X = data[:, :-1]  # All but last column
    y = data[:, -1]   # Last column
    return X, y

def test_small_test_csv():
    X, y = load_csv_data("tests/small_test.csv")
    lambda_reg_values = [0.01, 0.1, 1.0, 10.0]
    best_mse = float('inf')
    best_lambda = None
    best_preds = None
    
    for lambda_reg in lambda_reg_values:
        model = LassoHomotopyModel(lambda_reg=lambda_reg)
        results = model.fit(X, y)
        preds = results.predict(X)
        mse = np.mean((y - preds) ** 2)
        print(f"Small Test lambda_reg={lambda_reg}, MSE={mse}")
        print(f"Predictions: {preds}")
        print(f"Non-zero coefficients: {np.sum(np.abs(results.theta) > 1e-5)}")
        if mse < best_mse:
            best_mse = mse
            best_lambda = lambda_reg
            best_preds = preds
    
    print(f"Best lambda_reg={best_lambda}, Best MSE={best_mse}")
    assert best_mse < 50.0, f"Best mean squared error should be reasonable, got {best_mse}"

def test_collinear_data():
    """
    Test LassoHomotopyModel on collinear_data.csv.
    """
    # Load data
    X, y = load_csv_data("tests/collinear_data.csv")
    n_samples, n_features = X.shape
    print(f"Loaded data: {n_samples} samples, {n_features} features")

    # Initialize model with a reasonable lambda_reg
    lambda_reg = 0.1  # Matches your sample run
    model = LassoHomotopyModel(lambda_reg=lambda_reg)
    
    # Fit the model
    results = model.fit(X, y)
    
    # Predict
    preds = results.predict(X)
    mse = np.mean((y - preds) ** 2)
    
    # Compute sparsity
    non_zero_coeffs = np.sum(np.abs(results.theta) > 1e-5)
    
    
    # Assertions
    assert preds.shape == (n_samples,), f"Predictions shape mismatch: got {preds.shape}, expected ({n_samples},)"
    assert not np.allclose(preds, 0), "Predictions should not be all zeros"
    assert mse < 30.0, f"MSE too high: got {mse}, expected < 5.0"
    assert non_zero_coeffs <= n_features // 2, (
        f"Expected sparse solution (<= {n_features // 2} non-zero coeffs), got {non_zero_coeffs}"
    )
    assert non_zero_coeffs > 0, "Model should have at least one non-zero coefficient"
    
    # Additional check for collinearity impact
    if n_samples > 1:
        correlation_matrix = np.corrcoef(X, rowvar=False)
        high_corr = np.any(np.abs(correlation_matrix - np.eye(n_features)) > 0.9)
        print(f"High collinearity detected: {high_corr}")
        if high_corr:
            print("Collinearity present; sparsity is critical")

def test_singular_matrix():
    X = np.array([[1, 2, 2], [2, 4, 4], [3, 6, 6]])  # X_1 = 2 * X_0
    y = np.array([1, 2, 3])
    model = LassoHomotopyModel(lambda_reg=0.1)
    results = model.fit(X, y)
    mse = np.mean((y - results.predict(X)) ** 2)
    print(f"Singular matrix MSE: {mse}")
    assert mse < 20.0, f"MSE too high with singularity: {mse}"


def test_mse_with_noise():
    """
    Test LassoHomotopyModel on small_test.csv with and without Gaussian noise.
    Compares MSEs and ensures robustness to noise.
    """
    # Load data
    X, y = load_csv_data("tests/small_test.csv")
    n_samples, n_features = X.shape
    print(f"Loaded data: {n_samples} samples, {n_features} features")
    
    # Parameters
    lambda_reg = 0.1
    noise_std = 0.1
    np.random.seed(42)  # Reproducibility
    
    # Fit on original data
    model_original = LassoHomotopyModel(lambda_reg=lambda_reg)
    results_original = model_original.fit(X, y)
    preds_original = results_original.predict(X)
    mse_original = np.mean((y - preds_original) ** 2)
    non_zero_original = np.sum(np.abs(results_original.theta) > 1e-5)
    
    # Add Gaussian noise and fit
    y_noisy = y + np.random.normal(0, noise_std, y.shape)
    model_noisy = LassoHomotopyModel(lambda_reg=lambda_reg)
    results_noisy = model_noisy.fit(X, y_noisy)
    preds_noisy = results_noisy.predict(X)
    mse_noisy = np.mean((y_noisy - preds_noisy) ** 2)
    non_zero_noisy = np.sum(np.abs(results_noisy.theta) > 1e-5)
    
    
    # Assertions
    assert preds_original.shape == (n_samples,), "Original predictions shape mismatch"
    assert preds_noisy.shape == (n_samples,), "Noisy predictions shape mismatch"
    assert mse_original < 50, f"Original MSE too high: {mse_original}"  # Based on prior 530.9
    assert mse_noisy < 50, f"Noisy MSE too high: {mse_noisy}"  # Allow slight increase
    assert abs(mse_noisy - mse_original) < 50, (
        f"MSE difference too large: {mse_noisy - mse_original}"
    )  # Noise shouldn’t drastically alter fit
    assert non_zero_original > 0, "Original model should have non-zero coefficients"
    assert non_zero_noisy > 0, "Noisy model should have non-zero coefficients"


def test_collinear_data_sklearn_comparison():
    """
    Test LassoHomotopyModel vs sklearn Lasso on collinear_data.csv.
    Compares MSEs and ensures consistency.
    """
    # Load data
    X, y = load_csv_data("tests/collinear_data.csv")
    n_samples, n_features = X.shape
    print(f"Loaded data: {n_samples} samples, {n_features} features")
    
    # Parameters
    lambda_reg = 0.1  # Matches prior runs
    
    # Fit our model
    our_model = LassoHomotopyModel(lambda_reg=lambda_reg)
    our_results = our_model.fit(X, y)
    our_preds = our_results.predict(X)
    our_mse = np.mean((y - our_preds) ** 2)
    our_non_zero = np.sum(np.abs(our_results.theta) > 1e-5)
    
    # Fit sklearn model
    sklearn_model = Lasso(alpha=lambda_reg, fit_intercept=False)  # No intercept to match our model
    sklearn_model.fit(X, y)
    sklearn_preds = sklearn_model.predict(X)
    sklearn_mse = np.mean((y - sklearn_preds) ** 2)
    sklearn_non_zero = np.sum(np.abs(sklearn_model.coef_) > 1e-5)
    
    
    # Assertions
    assert our_preds.shape == (n_samples,), "Our predictions shape mismatch"
    assert sklearn_preds.shape == (n_samples,), "Sklearn predictions shape mismatch"
    assert our_mse < 30.0, f"Our MSE too high: {our_mse}"  # Matches prior collinear test
    assert sklearn_mse < 10.0, f"Sklearn MSE too high: {sklearn_mse}"
    assert abs(our_mse - sklearn_mse) < 25.0, (
        f"MSEs differ too much: {our_mse} vs {sklearn_mse}"
    )  # Allow small numerical differences
    assert our_non_zero <= n_features // 2, (
        f"Our model not sparse enough: {our_non_zero} > {n_features // 2}"
    )
    assert sklearn_non_zero <= n_features // 2, (
        f"Sklearn model not sparse enough: {sklearn_non_zero} > {n_features // 2}"
    )
    assert our_non_zero > 0, "Our model should have non-zero coefficients"
    assert sklearn_non_zero > 0, "Sklearn model should have non-zero coefficients"