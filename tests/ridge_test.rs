use coal_mind_linear_regression::base::{Matrix, Vector};
use coal_mind_linear_regression::{base, ridge};

/// Tests that Ridge with α=0 approximates OLS results
#[test]
fn test_ridge_zero_alpha() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect y = 2x relationship

    let features = Matrix::from_vec(5, 1, x);
    let targets = Vector::from_vec(y);

    // Check OLS first
    let ols_model = base::fit(&features, &targets).unwrap();
    assert!(
        (ols_model.coefficients[0] - 2.0).abs() < 1e-10,
        "OLS coefficient should be 2.0, got {}",
        ols_model.coefficients[0]
    );
    assert!(
        ols_model.intercept.abs() < 1e-10,
        "OLS intercept should be 0.0, got {}",
        ols_model.intercept
    );

    // Check Ridge with α=0
    let ridge_model = ridge::fit(&features, &targets, 0.0, 1000, 1e-8).unwrap();
    assert!(
        (ridge_model.coefficients[0] - 2.0).abs() < 1e-3,
        "Ridge coefficient should be close to 2.0, got {}",
        ridge_model.coefficients[0]
    );
    assert!(
        ridge_model.intercept.abs() < 1e-3,
        "Ridge intercept should be close to 0.0, got {}",
        ridge_model.intercept
    );

    // Compare predictions
    let x_test = Matrix::from_vec(3, 1, vec![6.0, 7.0, 8.0]);
    let ols_preds = base::predict(&ols_model, &x_test).unwrap();
    let ridge_preds = base::predict(&ridge_model, &x_test).unwrap();

    let pred_diff = (&ols_preds - &ridge_preds).norm();
    assert!(
        pred_diff < 1e-3,
        "Prediction difference too large: {}",
        pred_diff
    );
}

/// Tests that Ridge shrinks coefficients but doesn't make them exactly zero
#[test]
fn test_ridge_shrinkage() {
    let features = Matrix::from_vec(
        10,
        3,
        vec![
            1.0, 0.1, 0.0, 1.2, 0.2, 0.0, 0.8, 0.3, 0.0, 1.1, 0.15, 0.0, 0.9, 0.25, 0.0, 1.0, 0.2,
            0.0, 1.3, 0.1, 0.0, 0.7, 0.3, 0.0, 1.1, 0.15, 0.0, 0.9, 0.25, 0.0,
        ],
    );
    let targets = Vector::from_vec(vec![2.0, 2.4, 1.6, 2.2, 1.8, 2.0, 2.6, 1.4, 2.2, 1.8]);

    // Fit with high alpha
    let model = ridge::fit(&features, &targets, 1.0, 1000, 1e-6).unwrap();

    // Count non-zero coefficients (should be all of them for ridge)
    let non_zero = model
        .coefficients
        .iter()
        .filter(|&&x| x.abs() > 1e-10)
        .count();

    // Ridge should keep all coefficients non-zero
    assert_eq!(non_zero, model.coefficients.len());
}

/// Tests Ridge's behavior with different alpha values
/// Coefficients should decrease as alpha increases
#[test]
fn test_ridge_regularization_strength() {
    let features = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]);
    let targets = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let alphas = vec![0.0, 0.1, 0.5, 1.0];
    let mut coef_norms = Vec::new();

    for alpha in alphas {
        let model = ridge::fit(&features, &targets, alpha, 1000, 1e-6).unwrap();
        coef_norms.push(model.coefficients.norm());
    }

    // Check that coefficient norms decrease with increasing alpha
    for i in 1..coef_norms.len() {
        assert!(
            coef_norms[i] < coef_norms[i - 1],
            "Coefficient norm should decrease with higher alpha"
        );
    }
}

/// Tests Ridge's handling of multicollinearity
#[test]
fn test_ridge_multicollinearity() {
    // Create features where first two columns are perfectly correlated
    // but with a simpler relationship to target
    let features = Matrix::from_vec(
        10,
        2, // Reduced to 2 features
        vec![
            1.0, 1.0, // Perfectly correlated columns
            2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0,
            10.0,
        ],
    );

    // Target is simple linear combination plus some noise
    let targets = Vector::from_vec(vec![
        2.0, // roughly x1 + x2 + noise
        4.1, 6.0, 8.2, 10.1, 11.9, 14.0, 16.1, 18.0, 20.2,
    ]);

    // Use the best alpha we found
    let alpha = 0.1;
    let ridge_model = ridge::fit(&features, &targets, alpha, 1000, 1e-6).unwrap();
    let predictions = base::predict(&ridge_model, &features).unwrap();
    let r2 = base::score(&predictions, &targets).unwrap();

    // Check predictions match targets reasonably well
    assert!(r2 > 0.90, "Ridge R² score should be high, got {}", r2);

    // Check coefficient properties
    let coef_diff = (ridge_model.coefficients[0] - ridge_model.coefficients[1]).abs();
    assert!(
        coef_diff < 3.0,
        "Correlated features should have similar coefficients, difference was {}",
        coef_diff
    );

    // Both coefficients should be positive
    assert!(
        ridge_model.coefficients[0] > 0.0 && ridge_model.coefficients[1] > 0.0,
        "Coefficients should be positive"
    );
}

/// Tests error handling for invalid inputs
#[test]
fn test_ridge_input_validation() {
    let features = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let targets = Vector::from_vec(vec![1.0, 2.0]); // Wrong length

    // Test negative alpha
    let result1 = ridge::fit(&features, &targets, -1.0, 1000, 1e-6);
    assert!(result1.is_err());

    // Test dimension mismatch
    let result2 = ridge::fit(&features, &targets, 0.1, 1000, 1e-6);
    assert!(result2.is_err());
}

/// Tests prediction functionality with Ridge model
#[test]
fn test_ridge_predict() {
    let features = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
    let targets = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let model = ridge::fit(&features, &targets, 0.01, 1000, 1e-6).unwrap();
    let predictions = base::predict(&model, &features).unwrap();
    let rmse = base::rmse(&predictions, &targets).unwrap();

    assert!(rmse < 0.1, "RMSE too high: {}", rmse);
}

/// Tests reproducibility with same input
#[test]
fn test_ridge_reproducibility() {
    let features = Matrix::from_vec(4, 2, vec![1.0, 0.1, 2.0, 0.2, 3.0, 0.3, 4.0, 0.4]);
    let targets = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

    let model1 = ridge::fit(&features, &targets, 0.1, 1000, 1e-6).unwrap();
    let model2 = ridge::fit(&features, &targets, 0.1, 1000, 1e-6).unwrap();

    let coef_diff = (&model1.coefficients - &model2.coefficients).norm();
    assert!(
        coef_diff < 1e-10,
        "Models should be identical for same input"
    );
}

/// Tests that Ridge handles nearly singular matrices better than OLS
#[test]
fn test_ridge_near_singular() {
    let features = Matrix::from_vec(
        5,
        2,
        vec![
            1.0, 1.0001, // Nearly identical columns
            2.0, 2.0002, 3.0, 3.0003, 4.0, 4.0004, 5.0, 5.0005,
        ],
    );
    let targets = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    // OLS might fail or give unstable results
    let ridge_result = ridge::fit(&features, &targets, 0.1, 1000, 1e-6);
    assert!(
        ridge_result.is_ok(),
        "Ridge should handle near-singular matrix"
    );
}

#[test]
fn test_ridge_core_properties() {
    // Training data with EXPLICIT verification
    let train_data = vec![
        1.0, 0.5, // x₁, x₂
        2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5, 6.0, 3.0, 7.0, 3.5, 8.0, 4.0,
    ];

    let features = Matrix::from_vec(8, 2, train_data);

    // Target follows y = 2x₁ + 0.5x₂ pattern exactly
    let targets = features
        .row_iter()
        .map(|row| 2.0 * row[0] + 0.5 * row[1])
        .collect::<Vec<f64>>();

    let targets = Vector::from_vec(targets);

    // Test shrinkage with different alphas
    let alphas = vec![0.0, 0.1, 1.0, 10.0];
    let mut coefficient_norms = Vec::new();

    for alpha in &alphas {
        let model = ridge::fit(&features, &targets, *alpha, 1000, 1e-6).unwrap();
        let norm = model.coefficients.norm();
        coefficient_norms.push(norm);
    }

    // Verify coefficient shrinkage
    for i in 1..coefficient_norms.len() {
        assert!(
            coefficient_norms[i] < coefficient_norms[i - 1],
            "Coefficients should shrink with increasing alpha"
        );
    }

    // Test points that interpolate between training points
    let test_features = Matrix::from_vec(
        4,
        2,
        vec![
            1.5, 0.75, // Halfway between training points
            2.5, 1.25, 3.5, 1.75, 4.5, 2.25,
        ],
    );

    // Expected values using same relationship
    let test_targets = test_features
        .row_iter()
        .map(|row| 2.0 * row[0] + 0.5 * row[1])
        .collect::<Vec<f64>>();
    let test_targets = Vector::from_vec(test_targets);

    let ridge_model = ridge::fit(&features, &targets, 0.1, 1000, 1e-6).unwrap();
    let predictions = base::predict(&ridge_model, &test_features).unwrap();

    let rmse = base::rmse(&predictions, &test_targets).unwrap();

    assert!(
        rmse < 1.0,
        "RMSE should be reasonable for noisy data, got {}",
        rmse
    );

    // Test solution stability
    let model1 = ridge::fit(&features, &targets, 1.0, 1000, 1e-6).unwrap();
    let model2 = ridge::fit(&features, &targets, 1.0, 1000, 1e-6).unwrap();

    let coef_diff = (&model1.coefficients - &model2.coefficients).norm();
    assert!(
        coef_diff < 1e-10,
        "Solutions should be stable for same input"
    );
}
