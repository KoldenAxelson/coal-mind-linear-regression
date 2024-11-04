//! Ridge (L2) Regression Implementation
//!
//! Mathematical Background:
//! Ridge regression minimizes: (1/2n) ||y - Xβ||² + α||β||²
//! where:
//! - ||y - Xβ||² is the squared error loss
//! - ||β||² is the L2 norm of the coefficients squared
//! - α is the regularization strength
//!
//! Key Features:
//! - L2 regularization for coefficient shrinkage
//! - Improved stability for multicollinear features
//! - Closed-form solution via regularized normal equations
//! - Automatic standardization of features

use crate::base::{
    LinearRegressionError, LinearRegressionModel, LinearRegressionResult, Matrix, Vector,
};

/// Standardizes features and centers targets for ridge regression
///
/// # Mathematical Background
/// Standardization is important for ridge regression because:
/// 1. L2 penalty is scale-dependent
/// 2. Features must be on same scale for fair regularization
/// 3. Improves numerical stability
///
/// Process:
/// 1. Center features: x_centered = x - mean(x)
/// 2. Scale features: x_scaled = x_centered / std(x)
/// 3. Center targets: y_centered = y - mean(y)
///
/// # Arguments
/// * `features` - Input feature matrix
/// * `targets` - Target values vector
///
/// # Returns
/// * Tuple containing:
///   - Standardized feature matrix
///   - Feature means vector
///   - Feature standard deviations vector
///   - Target mean
///   - Centered target vector
pub(crate) fn standardize_data(
    features: &Matrix,
    targets: &Vector,
) -> (Matrix, Vector, Vector, f64, Vector) {
    let n_samples = features.nrows();
    let n_features = features.ncols();

    let mut feature_means = Vector::zeros(n_features);
    let mut feature_stds = Vector::zeros(n_features);
    let mut scaled_features = features.clone();

    // Calculate means and center features
    for j in 0..n_features {
        feature_means[j] = features.column(j).mean();
        for i in 0..n_samples {
            scaled_features[(i, j)] -= feature_means[j];
        }
    }

    // Calculate standard deviations and scale
    for j in 0..n_features {
        let col = scaled_features.column(j);
        let ss = col.dot(&col);
        feature_stds[j] = (ss / (n_samples as f64 - 1.0)).sqrt();
        if feature_stds[j] < 1e-10 {
            feature_stds[j] = 1.0;
        }
    }

    // Scale to unit variance
    for i in 0..n_samples {
        for j in 0..n_features {
            scaled_features[(i, j)] /= feature_stds[j];
        }
    }

    let target_mean = targets.mean();
    let centered_targets = targets - Vector::from_element(n_samples, target_mean);

    (
        scaled_features,
        feature_means,
        feature_stds,
        target_mean,
        centered_targets,
    )
}

/// Solves the regularized normal equations for ridge regression
///
/// # Mathematical Background
/// Ridge solution: β = (X^T X + αI)^(-1) X^T y
/// where:
/// - X is the feature matrix
/// - y is the target vector
/// - α is the regularization parameter
/// - I is the identity matrix
///
/// # Arguments
/// * `x` - Feature matrix (usually standardized)
/// * `y` - Target vector (usually centered)
/// * `alpha` - Regularization strength
///
/// # Returns
/// * Result containing either:
///   - Vector of coefficients (β)
///   - Error message if matrix inversion fails
fn solve_ridge_equation(x: &Matrix, y: &Vector, alpha: f64) -> Result<Vector, String> {
    let gram_matrix = x.transpose() * x;
    let n_features = x.ncols();
    let identity = Matrix::identity(n_features, n_features);
    let regularized_gram = &gram_matrix + alpha * identity;

    match regularized_gram.try_inverse() {
        Some(gram_inv) => {
            let x_t = x.transpose();
            let rhs = &x_t * y;
            let result = &gram_inv * rhs;
            Ok(Vector::from_column_slice(result.as_slice()))
        }
        None => Err("Failed to solve ridge equation: singular matrix".to_string()),
    }
}

/// Fits a ridge regression model using the normal equation method
///
/// # Mathematical Background
/// Ridge regression optimization problem:
/// min (1/2n)||y - Xβ||² + α||β||²
///
/// Process:
/// 1. Standardize features and center targets
/// 2. Solve (X^T X + αI)β = X^T y
/// 3. Transform coefficients back to original scale
///
/// # Arguments
/// * `features` - Feature matrix X where each row is an observation
/// * `targets` - Target vector y
/// * `alpha` - L2 regularization strength (must be non-negative)
/// * `max_iter` - Unused (kept for API compatibility)
/// * `tol` - Unused (kept for API compatibility)
///
/// # Returns
/// * Result containing either:
///   - Fitted LinearRegressionModel
///   - Error if dimensions mismatch or alpha is negative
///
/// # Notes
/// - Higher α values produce more shrinkage
/// - For α=0, equivalent to standard linear regression
/// - Features are standardized internally
pub fn fit(
    features: &Matrix,
    targets: &Vector,
    alpha: f64,
    _max_iter: usize,
    _tol: f64,
) -> LinearRegressionResult<LinearRegressionModel> {
    if alpha == 0.0 {
        return crate::base::fit(features, targets);
    }

    let (n_samples, _) = features.shape();

    if alpha < 0.0 {
        return Err(LinearRegressionError(
            "Alpha parameter must be non-negative".to_string(),
        ));
    }

    if n_samples != targets.len() {
        return Err(LinearRegressionError(
            "Features and targets dimension mismatch".to_string(),
        ));
    }

    let (scaled_x, x_means, x_stds, y_mean, centered_y) = standardize_data(features, targets);

    let beta = match solve_ridge_equation(&scaled_x, &centered_y, alpha) {
        Ok(b) => b,
        Err(e) => return Err(LinearRegressionError(e)),
    };

    let mut coefficients = beta;
    for j in 0..coefficients.len() {
        coefficients[j] /= x_stds[j];
    }

    let intercept = y_mean - x_means.dot(&coefficients);

    Ok(LinearRegressionModel {
        coefficients,
        intercept,
    })
}
