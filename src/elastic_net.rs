//! Elastic Net Regression Implementation
//!
//! Mathematical Background:
//! Elastic Net combines L1 and L2 regularization, minimizing:
//! (1/2n) ||y - Xβ||² + α[ρ||β||₁ + (1-ρ)||β||²/2]
//! where:
//! - ||y - Xβ||² is the squared error loss
//! - ||β||₁ is the L1 norm (sum of absolute values)
//! - ||β||² is the L2 norm squared
//! - α is the regularization strength
//! - ρ (l1_ratio) balances L1 vs L2 regularization
//!
//! Key Features:
//! - Combines benefits of Lasso (sparsity) and Ridge (grouped selection)
//! - Handles correlated features better than Lasso alone
//! - Provides sparse solutions when ρ approaches 1
//! - Degrades gracefully to Ridge regression as ρ approaches 0
//! - Uses coordinate descent optimization
//! - Automatically standardizes features

use crate::base::{
    LinearRegressionError, LinearRegressionModel, LinearRegressionResult, Matrix, Vector,
};
use crate::lasso::soft_threshold;
use crate::ridge::standardize_data;

/// Fits an Elastic Net regression model using coordinate descent
///
/// # Mathematical Background
/// Elastic Net optimization problem:
/// min (1/2n)||y - Xβ||² + α[ρ||β||₁ + (1-ρ)||β||²/2]
///
/// Process:
/// 1. Standardize features and center targets
/// 2. Initialize β = 0
/// 3. For each feature j until convergence:
///    - Calculate partial residual r = y - Xβ + xⱼβⱼ
///    - Update βⱼ using combined L1 and L2 penalties
///    - Apply soft thresholding for L1 component
/// 4. Transform coefficients back to original scale
///
/// # Arguments
/// * `features` - Feature matrix X where each row is an observation
/// * `targets` - Target vector y
/// * `alpha` - Overall regularization strength (must be non-negative)
/// * `l1_ratio` - Balance between L1 and L2 regularization (ρ)
///   - 0: Ridge regression
///   - 1: Lasso regression
///   - (0,1): Elastic Net
/// * `max_iter` - Maximum number of coordinate descent iterations
/// * `tol` - Convergence tolerance for coordinate descent
///
/// # Returns
/// * Result containing either:
///   - Fitted LinearRegressionModel
///   - Error if inputs are invalid
///
/// # Notes
/// - For α=0 or l1_ratio outside [0,1], defaults to standard linear regression
/// - Features are standardized internally for numerical stability
/// - L2 penalty is scaled by 0.5 to match standard formulation
/// - Coordinate descent includes feature-wise normalization
///
/// # Example
/// ```no_run
/// use coal_mind_linear_regression::{elastic_net, base::{Matrix, Vector, LinearRegressionResult}};
///
/// let features = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let targets = Vector::from_vec(vec![2.0, 4.0, 6.0]);
/// let model: LinearRegressionResult<_> = elastic_net::fit(&features, &targets, 0.5, 0.5, 1000, 1e-6);
/// ```
pub fn fit(
    features: &Matrix,
    targets: &Vector,
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
) -> LinearRegressionResult<LinearRegressionModel> {
    // [Rest of implementation stays exactly the same]
    // Default to standard linear regression for edge cases
    if alpha == 0.0 || !(0.0..=1.0).contains(&l1_ratio) {
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

    // Standardize data for numerical stability
    let (scaled_x, x_means, x_stds, y_mean, centered_y) = standardize_data(features, targets);

    // Initialize coefficient vector
    let mut beta = Vector::zeros(n_features);
    let mut beta_old;

    // Calculate regularization strengths
    // L2 penalty scaled by 0.5 to match standard formulation
    let l1_strength = alpha * l1_ratio;
    let l2_strength = alpha * (1.0 - l1_ratio) * 0.5;

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

            // Calculate update terms
            let x_j_sq = x_j.dot(&x_j) / (n_samples as f64);
            let r_x_j = x_j.dot(&r) / (n_samples as f64);

            // Update coefficient with combined L1/L2 regularization
            let z_j = r_x_j;
            if x_j_sq == 0.0 {
                beta[j] = 0.0;
            } else {
                // Ridge update followed by Lasso soft thresholding
                let beta_j = z_j / (x_j_sq + l2_strength);
                beta[j] = soft_threshold(beta_j, l1_strength / x_j_sq);
            }
        }

        // Check convergence
        if (&beta - &beta_old).norm() < tol {
            break;
        }
    }

    // Transform coefficients back to original scale
    let mut coefficients = beta;
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
