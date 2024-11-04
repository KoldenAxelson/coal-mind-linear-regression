use nalgebra as na;

/// Type aliases for improved readability
pub type Matrix = na::DMatrix<f64>;
pub type Vector = na::DVector<f64>;
pub type LinearRegressionResult<T> = Result<T, LinearRegressionError>;

/// Custom error type for Linear Regression operations
#[derive(Debug, Clone)]
pub struct LinearRegressionError(pub String);

/// Represents a fitted Linear Regression model
/// Contains the computed coefficients (β) and intercept (β₀)
#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    /// Model coefficients (β) for each feature
    pub coefficients: Vector,
    /// Model intercept (β₀)
    pub intercept: f64,
}

/// Centers the data by subtracting means from features and targets
///
/// # Arguments
/// * `features` - Input feature matrix where each column is a feature
/// * `targets` - Target values vector
///
/// # Returns
/// * Tuple containing:
///   - centered feature matrix
///   - feature means vector
///   - target mean scalar
///   - centered target vector
///
/// # Mathematical Context
/// Centering the data improves numerical stability and simplifies intercept calculation.
/// For each feature j: x_centered[i,j] = x[i,j] - mean(x[*,j])
/// For targets: y_centered[i] = y[i] - mean(y)
fn center_data(features: &Matrix, targets: &Vector) -> (Matrix, Vector, f64, Vector) {
    // Calculate mean of each feature column
    let feature_means: Vec<f64> = (0..features.ncols())
        .map(|j| features.column(j).mean())
        .collect();
    let y_mean = targets.mean();

    // Center features by subtracting column means
    let mut centered_x = features.clone();
    for j in 0..features.ncols() {
        for i in 0..features.nrows() {
            centered_x[(i, j)] -= feature_means[j];
        }
    }

    // Center targets by subtracting mean
    let centered_y = targets - Vector::from_element(targets.len(), y_mean);

    (
        centered_x,
        Vector::from_vec(feature_means),
        y_mean,
        centered_y,
    )
}

/// Solves the normal equation (X^T * X)β = X^T * y for β
///
/// # Arguments
/// * `x` - Feature matrix (usually centered)
/// * `y` - Target vector (usually centered)
///
/// # Returns
/// * Result containing either:
///   - Vector of coefficients (β)
///   - Error message if matrix is singular
///
/// # Mathematical Context
/// The normal equation minimizes the sum of squared residuals:
/// β = (X^T * X)^(-1) * X^T * y
fn solve_normal_equation(x: &Matrix, y: &Vector) -> Result<Vector, String> {
    // Compute Gram matrix (X^T * X)
    let gram_matrix = x.transpose() * x;

    match gram_matrix.try_inverse() {
        Some(gram_inv) => {
            let x_t = x.transpose();
            let rhs = &x_t * y;
            let result = &gram_inv * rhs;
            Ok(Vector::from_column_slice(result.as_slice()))
        }
        None => Err("Failed to solve normal equation: singular matrix".to_string()),
    }
}

/// Calculates the intercept using feature means and coefficients
///
/// # Arguments
/// * `feature_means` - Vector of feature means
/// * `y_mean` - Target mean
/// * `coefficients` - Computed coefficients (β)
///
/// # Returns
/// * Intercept value (β₀)
///
/// # Mathematical Context
/// After solving for β with centered data, calculate β₀:
/// β₀ = ȳ - Σ(x̄ᵢβᵢ)
fn calculate_intercept(feature_means: &Vector, y_mean: f64, coefficients: &Vector) -> f64 {
    assert_eq!(
        feature_means.len(),
        coefficients.len(),
        "feature_means and coefficients must have same length"
    );

    let mean_contribution = feature_means.dot(coefficients);
    y_mean - mean_contribution
}

/// Fits a linear regression model using the normal equation method
///
/// # Arguments
/// * `features` - Matrix X where each row is an observation and each column is a feature
/// * `targets` - Vector y of target values
///
/// # Returns
/// * Result containing either:
///   - Fitted LinearRegressionModel
///   - Error if dimensions mismatch or matrix is singular
///
/// # Mathematical Context
/// 1. Centers the data
/// 2. Solves (X^T * X)β = X^T * y for β
/// 3. Calculates intercept β₀ = ȳ - Σ(x̄ᵢβᵢ)
pub fn fit(features: &Matrix, targets: &Vector) -> LinearRegressionResult<LinearRegressionModel> {
    let (num_samples, _) = features.shape();

    if num_samples != targets.len() {
        return Err(LinearRegressionError(format!(
            "Features and targets shape mismatch: features {} rows != targets {} elements",
            num_samples,
            targets.len()
        )));
    }

    let (centered_x, feature_means, y_mean, centered_y) = center_data(features, targets);

    let coefficients = match solve_normal_equation(&centered_x, &centered_y) {
        Ok(coef) => coef,
        Err(e) => return Err(LinearRegressionError(e)),
    };

    let intercept = calculate_intercept(&feature_means, y_mean, &coefficients);

    Ok(LinearRegressionModel {
        coefficients,
        intercept,
    })
}

/// Makes predictions using a fitted linear regression model
///
/// # Arguments
/// * `model` - Fitted LinearRegressionModel
/// * `features` - Feature matrix for prediction
///
/// # Returns
/// * Result containing either:
///   - Vector of predictions
///   - Error if dimensions mismatch
///
/// # Mathematical Context
/// Computes ŷ = Xβ + β₀
pub fn predict(model: &LinearRegressionModel, features: &Matrix) -> LinearRegressionResult<Vector> {
    if features.ncols() != model.coefficients.len() {
        return Err(LinearRegressionError(format!(
            "Feature dimension mismatch: expected {}, got {}",
            model.coefficients.len(),
            features.ncols()
        )));
    }

    let predictions = features * &model.coefficients;
    let intercept_vec = Vector::from_element(features.nrows(), model.intercept);
    Ok(predictions + intercept_vec)
}

/// Calculates the R² score (coefficient of determination)
///
/// # Arguments
/// * `predictions` - Model predictions (ŷ)
/// * `actual` - Actual target values (y)
///
/// # Returns
/// * Result containing either:
///   - R² score between 0 and 1
///   - Error if dimensions mismatch
///
/// # Mathematical Context
/// R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
/// Represents the proportion of variance in the target variable
/// that is predictable from the features
pub fn score(predictions: &Vector, actual: &Vector) -> LinearRegressionResult<f64> {
    if predictions.len() != actual.len() {
        return Err(LinearRegressionError(format!(
            "Length mismatch: predictions {} != actual {}",
            predictions.len(),
            actual.len()
        )));
    }

    let residual_sum = (predictions - actual).map(|x| x * x).sum();
    let actual_mean = actual.mean();
    let mean_vec = Vector::from_element(actual.len(), actual_mean);
    let total_sum = (actual - mean_vec).map(|x| x * x).sum();

    Ok(1.0 - (residual_sum / total_sum))
}

/// Calculates the Root Mean Square Error
///
/// # Arguments
/// * `predictions` - Model predictions (ŷ)
/// * `actual` - Actual target values (y)
///
/// # Returns
/// * Result containing either:
///   - RMSE value
///   - Error if dimensions mismatch
///
/// # Mathematical Context
/// RMSE = √(Σ(y - ŷ)² / n)
/// Measures the average magnitude of prediction errors
pub fn rmse(predictions: &Vector, actual: &Vector) -> LinearRegressionResult<f64> {
    if predictions.len() != actual.len() {
        return Err(LinearRegressionError(format!(
            "Length mismatch: predictions {} != actual {}",
            predictions.len(),
            actual.len()
        )));
    }

    let mse = (predictions - actual).map(|x| x * x).mean();
    Ok(mse.sqrt())
}
