// tests/basic_test.rs

use coal_mind_linear_regression::base::{self, Matrix, Vector};

/// Tests fitting and prediction with perfect linear relationship
/// y = 2x
#[test]
fn test_perfect_linear_relationship() {
    let features = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
    let targets = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let model = base::fit(&features, &targets).unwrap();

    // Model should have 1 coefficient (1 feature)
    assert_eq!(model.coefficients.len(), 1);

    let predictions = base::predict(&model, &features).unwrap();

    // Predictions should have same length as targets
    assert_eq!(predictions.len(), targets.len());

    let expected = Vector::from_vec(vec![2.0, 4.0, 6.0]);
    assert!((predictions - expected).norm() < 1e-10);
}

/// Tests R² score with perfect predictions
/// Expects a score of 1.0
#[test]
fn test_perfect_r2_score() {
    let predictions = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let actual = Vector::from_vec(vec![1.0, 2.0, 3.0]);

    let r2_score = base::score(&predictions, &actual).unwrap();
    assert!((r2_score - 1.0).abs() < 1e-10);
}

/// Tests RMSE calculation with slightly imperfect predictions
/// Ensures error is within expected range
#[test]
fn test_rmse_calculation() {
    let predictions = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let actual = Vector::from_vec(vec![1.1, 2.1, 2.9]);

    let error = base::rmse(&predictions, &actual).unwrap();
    assert!(error > 0.0 && error < 0.15);
}

/// Tests error handling for dimension mismatch during fitting
#[test]
fn test_fit_dimension_mismatch() {
    let features = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let targets = Vector::from_vec(vec![1.0, 2.0]); // Wrong number of targets

    assert!(base::fit(&features, &targets).is_err());
}

/// Tests error handling for dimension mismatch during prediction
#[test]
fn test_predict_dimension_mismatch() {
    // Train with 1D features
    let training_features = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
    let targets = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let model = base::fit(&training_features, &targets).unwrap();

    // Try to predict with 2D features
    let test_features = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(base::predict(&model, &test_features).is_err());
}

/// Tests model performance with noisy linear data
#[test]
fn test_noisy_linear_fit() {
    let features = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let targets = Vector::from_vec(vec![2.1, 3.9, 6.2, 7.8]); // y ≈ 2x + noise

    let model = base::fit(&features, &targets).unwrap();
    let predictions = base::predict(&model, &features).unwrap();

    let r2 = base::score(&predictions, &targets).unwrap();
    assert!(r2 > 0.95); // Strong but not perfect fit
}

/// Tests multivariate regression (multiple features)
#[test]
fn test_multivariate_regression() {
    let features = Matrix::from_vec(
        3,
        2,
        vec![
            1.0, 0.0, // First sample
            2.0, 1.0, // Second sample
            1.0, 2.0, // Third sample
        ],
    );
    let targets = Vector::from_vec(vec![2.0, 4.0, 5.0]); // y ≈ x1 + 1.5x2

    let model = base::fit(&features, &targets).unwrap();
    let predictions = base::predict(&model, &features).unwrap();

    let r2 = base::score(&predictions, &targets).unwrap();
    assert!(r2 > 0.99); // Should be nearly perfect
}

/// Tests handling of error cases in metric calculations
#[test]
fn test_metric_error_handling() {
    let predictions = Vector::from_vec(vec![1.0, 2.0]);
    let actual = Vector::from_vec(vec![1.0, 2.0, 3.0]); // Different lengths

    assert!(base::score(&predictions, &actual).is_err());
    assert!(base::rmse(&predictions, &actual).is_err());
}
