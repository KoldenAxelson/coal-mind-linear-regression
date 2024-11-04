//! Lasso (Least Absolute Shrinkage and Selection Operator) Regression Implementation
//!
//! Mathematical Background:
//! Lasso regression minimizes: (1/2n) ||y - Xβ||² + α||β||₁
//! where:
//! - ||y - Xβ||² is the squared error loss
//! - ||β||₁ is the L1 norm of the coefficients
//! - α is the regularization strength
//!
//! Key Features:
//! - Feature selection through L1 regularization
//! - Automatic handling of multicollinearity
//! - Sparse coefficient solutions for high α values
//! - Coordinate descent optimization

use crate::base::{
    LinearRegressionError, LinearRegressionModel, LinearRegressionResult, Matrix, Vector,
};

/// Implements the soft-thresholding operator for coordinate descent
///
/// # Mathematical Background
/// S(z, γ) = sign(z) * max(|z| - γ, 0)
/// where:
/// - z is the input value
/// - γ (gamma) is the threshold parameter
///
/// This operator is key to L1 regularization as it:
/// 1. Sets small coefficients exactly to zero
/// 2. Shrinks other coefficients toward zero
///
/// # Arguments
/// * `z` - Input value to be thresholded
/// * `gamma` - Threshold value (typically α in Lasso)
///
/// # Returns
/// * Thresholded value according to the soft-thresholding rule
pub(crate) fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

/// Standardizes features and centers targets for Lasso regression
///
/// # Mathematical Background
/// Standardization is crucial for Lasso because:
/// 1. L1 penalty is scale-dependent
/// 2. Features must be on same scale for fair regularization
/// 3. Improves numerical stability of coordinate descent
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
fn standardize_data(features: &Matrix, targets: &Vector) -> (Matrix, Vector, Vector, f64, Vector) {
    let n_samples = features.nrows();
    let n_features = features.ncols();

    let mut feature_means = Vector::zeros(n_features);
    let mut feature_stds = Vector::zeros(n_features);
    let mut scaled_features = features.clone();

    // Calculate feature means
    for j in 0..n_features {
        feature_means[j] = features.column(j).mean();
    }

    // Center features
    for i in 0..n_samples {
        for j in 0..n_features {
            scaled_features[(i, j)] -= feature_means[j];
        }
    }

    // Calculate feature standard deviations
    for j in 0..n_features {
        let col = scaled_features.column(j);
        let ss = col.dot(&col);
        // Use n-1 for unbiased estimation
        feature_stds[j] = (ss / (n_samples as f64 - 1.0)).sqrt();
        // Prevent division by zero for constant features
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

    // Center targets
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

/// Fits a Lasso regression model using coordinate descent
///
/// # Mathematical Background
/// Lasso optimization problem:
/// min (1/2n)||y - Xβ||² + α||β||₁
///
/// Coordinate descent algorithm:
/// 1. Initialize β = 0
/// 2. Until convergence:
///    For each feature j:
///    - Calculate partial residual r = y - Xβ + xⱼβⱼ
///    - Update βⱼ using soft thresholding
///
/// # Arguments
/// * `features` - Feature matrix X where each row is an observation
/// * `targets` - Target vector y
/// * `alpha` - L1 regularization strength (must be non-negative)
/// * `max_iter` - Maximum number of coordinate descent iterations
/// * `tol` - Convergence tolerance for coordinate descent
///
/// # Returns
/// * `LinearRegressionResult<LinearRegressionModel>` - Fitted model or error
///
/// # Notes
/// - Higher α values produce sparser solutions
/// - Features are standardized internally
/// - For α=0, delegates to standard linear regression
/// - Convergence is checked using coefficient changes
pub fn fit(
    features: &Matrix,
    targets: &Vector,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> LinearRegressionResult<LinearRegressionModel> {
    // Special case: α=0 is equivalent to standard linear regression
    if alpha == 0.0 {
        return crate::base::fit(features, targets);
    }

    let (n_samples, n_features) = features.shape();

    // Validate inputs
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

    // Standardize data for stable optimization
    let (scaled_x, x_means, x_stds, y_mean, centered_y) = standardize_data(features, targets);

    // Initialize coefficient vector to zero
    let mut beta = Vector::zeros(n_features);
    let mut beta_old;

    // Coordinate descent optimization
    for _ in 0..max_iter {
        beta_old = beta.clone();

        // Update each coefficient in turn
        for j in 0..n_features {
            let x_j = scaled_x.column(j);

            // Calculate partial residual excluding current feature
            let mut r = centered_y.clone();
            for k in 0..n_features {
                if k != j {
                    r -= beta[k] * scaled_x.column(k);
                }
            }

            // Update coefficient using soft thresholding
            let z_j = x_j.dot(&r) / (n_samples as f64);
            beta[j] = soft_threshold(z_j, alpha);
        }

        // Check convergence using coefficient changes
        if (&beta - &beta_old).norm() < tol {
            break;
        }
    }

    // Transform coefficients back to original scale
    let mut coefficients = beta.clone();
    for j in 0..n_features {
        coefficients[j] /= x_stds[j];
    }

    // Calculate intercept using means
    let intercept = y_mean - x_means.dot(&coefficients);

    Ok(LinearRegressionModel {
        coefficients,
        intercept,
    })
}
