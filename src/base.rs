use nalgebra as na;

/// Type aliases for improved readability and maintainability
pub type Matrix = na::DMatrix<f64>;
pub type Vector = na::DVector<f64>;
pub type LinearRegressionResult<T> = Result<T, LinearRegressionError>;

/// Custom error type for Linear Regression operations
/// Wraps error messages in a type-safe structure
#[derive(Debug, Clone)]
pub struct LinearRegressionError(String);

/// Represents a fitted Linear Regression model
/// Stores the computed coefficients (β) and intercept (β₀)
#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    /// Model coefficients (β) for each feature
    coefficients: Vector,
    /// Model intercept (β₀)
    intercept: f64,
}

/// Centers the data by subtracting the mean from features and targets
///
/// # Mathematical Background
/// Improves numerical stability and simplifies intercept calculation
///
/// # Arguments
/// * `features` - Input feature matrix
/// * `targets` - Target values
///
/// # Returns
/// * Tuple containing:
///   - centered feature matrix
///   - feature means vector
///   - target mean scalar
///   - centered target vector
fn center_data(features: &Matrix, targets: &Vector) -> (Matrix, Vector, f64, Vector) {
    let x_mean = features.column_mean();
    let y_mean = targets.mean();

    let x_mean_matrix =
        Matrix::from_columns(&vec![
            Vector::from_element(features.nrows(), x_mean[0]);
            features.ncols()
        ]);

    let centered_x = features - &x_mean_matrix;
    let centered_y = targets - Vector::from_element(targets.len(), y_mean);

    (centered_x, x_mean, y_mean, centered_y)
}

/// Solves the normal equation (X^T * X)β = X^T * y
///
/// # Mathematical Background
/// The normal equation minimizes the sum of squared residuals:
/// β = (X^T * X)^(-1) * X^T * y
///
/// # Arguments
/// * `x` - Feature matrix (usually centered)
/// * `y` - Target vector (usually centered)
///
/// # Returns
/// * Result containing either:
///   - Vector of coefficients (β)
///   - Error message if matrix is singular
fn solve_normal_equation(x: &Matrix, y: &Vector) -> Result<Vector, String> {
    let gram_matrix = x.transpose() * x;

    match gram_matrix.try_inverse() {
        Some(gram_inv) => Ok(gram_inv * x.transpose() * y),
        None => Err("Failed to solve normal equation: singular matrix".to_string()),
    }
}

/// Calculates the intercept using means and coefficients
///
/// # Mathematical Background
/// After solving for β with centered data, we calculate β₀:
/// β₀ = ȳ - x̄ᵀβ
///
/// # Arguments
/// * `x_mean` - Feature means vector
/// * `y_mean` - Target mean scalar
/// * `coefficients` - Computed coefficients (β)
fn calculate_intercept(x_mean: &Vector, y_mean: f64, coefficients: &Vector) -> f64 {
    let mean_contribution = (x_mean.transpose() * coefficients)[(0, 0)];
    y_mean - mean_contribution
}

/// Fits a linear regression model using the normal equation method
///
/// # Mathematical Background
/// Solves the equation (X^T * X)β = X^T * y for β
/// Uses mean centering for numerical stability
///
/// # Arguments
/// * `features` - Matrix X where each row is an observation and each column is a feature
/// * `targets` - Vector y of target values
///
/// # Returns
/// * `RegressionResult<LinearRegressionModel>` - Fitted model or error
pub fn fit(features: &Matrix, targets: &Vector) -> LinearRegressionResult<LinearRegressionModel> {
    let (num_samples, _) = features.shape();

    if num_samples != targets.len() {
        return Err(LinearRegressionError(format!(
            "Features and targets shape mismatch: features {} rows != targets {} elements",
            num_samples,
            targets.len()
        )));
    }

    let (centered_x, x_mean, y_mean, centered_y) = center_data(features, targets);

    let coefficients = match solve_normal_equation(&centered_x, &centered_y) {
        Ok(coef) => coef,
        Err(e) => return Err(LinearRegressionError(e)),
    };

    let intercept = calculate_intercept(&x_mean, y_mean, &coefficients);

    Ok(LinearRegressionModel {
        coefficients,
        intercept,
    })
}

/// Makes predictions using a fitted linear regression model
///
/// # Formula
/// ŷ = Xβ + β₀
///
/// # Arguments
/// * `model` - Fitted LinearRegressionModel
/// * `features` - Feature matrix for prediction
///
/// # Returns
/// * `RegressionResult<Vector>` - Predicted values or error
pub fn predict(model: &LinearRegressionModel, features: &Matrix) -> LinearRegressionResult<Vector> {
    if features.ncols() != model.coefficients.len() {
        return Err(LinearRegressionError(format!(
            "Feature dimension mismatch: expected {}, got {}",
            model.coefficients.len(),
            features.ncols()
        )));
    }

    let intercept_vec = Vector::from_element(features.nrows(), model.intercept);
    Ok(features * &model.coefficients + intercept_vec)
}

/// Calculates the R² score (coefficient of determination)
///
/// # Mathematical Background
/// R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
/// Represents the proportion of variance in the target variable explained by the model
///
/// # Arguments
/// * `predictions` - Model predictions
/// * `actual` - Actual target values
///
/// # Returns
/// * `RegressionResult<f64>` - Score between 0 and 1, or error
pub fn score(predictions: &Vector, actual: &Vector) -> LinearRegressionResult<f64> {
    if predictions.len() != actual.len() {
        return Err(LinearRegressionError(format!(
            "Length mismatch: predictions {} != actual {}",
            predictions.len(),
            actual.len()
        )));
    }

    let residual_sum = (predictions - actual).map(|x| x * x).sum();

    // Create a vector of means for subtraction
    let actual_mean = actual.mean();
    let mean_vec = Vector::from_element(actual.len(), actual_mean);
    let total_sum = (actual - mean_vec).map(|x| x * x).sum();

    Ok(1.0 - (residual_sum / total_sum))
}

/// Calculates the Root Mean Square Error
///
/// # Mathematical Background
/// RMSE = √(Σ(y - ŷ)² / n)
/// Measures the average magnitude of prediction errors
///
/// # Arguments
/// * `predictions` - Model predictions
/// * `actual` - Actual target values
///
/// # Returns
/// * `RegressionResult<f64>` - RMSE value or error
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
