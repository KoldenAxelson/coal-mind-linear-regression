use coal_mind_linear_regression::base::{Matrix, Vector};
use coal_mind_linear_regression::{base, elastic_net};

/// Tests that Elastic Net with α=0 or l1_ratio outside [0,1] defaults to OLS
#[test]
fn test_elastic_net_to_ols() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x relationship

    let features = Matrix::from_vec(5, 1, x);
    let targets = Vector::from_vec(y);

    let ols_model = base::fit(&features, &targets).unwrap();

    // Test alpha = 0
    let en_model1 = elastic_net::fit(&features, &targets, 0.0, 0.5, 1000, 1e-8).unwrap();

    // Test l1_ratio outside [0,1]
    let en_model2 = elastic_net::fit(&features, &targets, 0.1, -0.5, 1000, 1e-8).unwrap();
    let en_model3 = elastic_net::fit(&features, &targets, 0.1, 1.5, 1000, 1e-8).unwrap();

    let x_test = Matrix::from_vec(3, 1, vec![6.0, 7.0, 8.0]);
    let ols_preds = base::predict(&ols_model, &x_test).unwrap();
    let en_preds1 = base::predict(&en_model1, &x_test).unwrap();
    let en_preds2 = base::predict(&en_model2, &x_test).unwrap();
    let en_preds3 = base::predict(&en_model3, &x_test).unwrap();

    assert!((&ols_preds - &en_preds1).norm() < 1e-3);
    assert!((&ols_preds - &en_preds2).norm() < 1e-3);
    assert!((&ols_preds - &en_preds3).norm() < 1e-3);
}

/// Tests that Elastic Net with l1_ratio=1 approximates Lasso
#[test]
fn test_elastic_net_to_lasso() {
    let features = Matrix::from_vec(
        10,
        3,
        vec![
            1.0, 0.1, 0.0, 1.2, 0.2, 0.0, 0.8, 0.3, 0.0, 1.1, 0.15, 0.0, 0.9, 0.25, 0.0, 1.0, 0.2,
            0.0, 1.3, 0.1, 0.0, 0.7, 0.3, 0.0, 1.1, 0.15, 0.0, 0.9, 0.25, 0.0,
        ],
    );
    let targets = Vector::from_vec(vec![2.0, 2.4, 1.6, 2.2, 1.8, 2.0, 2.6, 1.4, 2.2, 1.8]);

    let alpha = 1.0;
    let model = elastic_net::fit(&features, &targets, alpha, 1.0, 1000, 1e-6).unwrap();

    // Count non-zero coefficients
    let non_zero = model
        .coefficients
        .iter()
        .filter(|&&x| x.abs() > 1e-10)
        .count();

    // Should have eliminated at least one feature (Lasso-like behavior)
    assert!(non_zero < model.coefficients.len());
}

/// Tests that Elastic Net with l1_ratio=0 approximates Ridge
#[test]
fn test_elastic_net_to_ridge() {
    let features = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]);
    let targets = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let model = elastic_net::fit(&features, &targets, 1.0, 0.0, 1000, 1e-6).unwrap();

    // All coefficients should be non-zero (Ridge-like behavior)
    let non_zero = model
        .coefficients
        .iter()
        .filter(|&&x| x.abs() > 1e-10)
        .count();
    assert_eq!(non_zero, model.coefficients.len());
}

/// Tests error handling for invalid inputs
#[test]
fn test_elastic_net_input_validation() {
    let features = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let targets = Vector::from_vec(vec![1.0, 2.0]); // Wrong length

    // Test negative alpha
    let result1 = elastic_net::fit(&features, &targets, -1.0, 0.5, 1000, 1e-6);
    assert!(result1.is_err());

    // Test dimension mismatch
    let result2 = elastic_net::fit(&features, &targets, 0.1, 0.5, 1000, 1e-6);
    assert!(result2.is_err());
}

/// Tests reproducibility
#[test]
fn test_elastic_net_reproducibility() {
    let features = Matrix::from_vec(4, 2, vec![1.0, 0.1, 2.0, 0.2, 3.0, 0.3, 4.0, 0.4]);
    let targets = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

    let model1 = elastic_net::fit(&features, &targets, 0.1, 0.5, 1000, 1e-6).unwrap();
    let model2 = elastic_net::fit(&features, &targets, 0.1, 0.5, 1000, 1e-6).unwrap();

    let coef_diff = (&model1.coefficients - &model2.coefficients).norm();
    assert!(
        coef_diff < 1e-10,
        "Models should be identical for same input"
    );
}

#[test]
fn test_elastic_net_coefficient_path() {
    let features = Matrix::from_vec(5, 2, vec![1.0, 0.5, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5]);
    let targets = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let l1_ratios = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let mut coef_norms = Vec::new();

    for l1_ratio in &l1_ratios {
        let model = elastic_net::fit(&features, &targets, 1.0, *l1_ratio, 1000, 1e-6).unwrap();
        let norm = model.coefficients.norm();
        coef_norms.push(norm);
    }

    for i in 1..coef_norms.len() {
        assert!(
            coef_norms[i] <= coef_norms[i - 1] * 1.1,
            "\nNorm increased from {} to {} at l1_ratio {} -> {}",
            coef_norms[i - 1],
            coef_norms[i],
            l1_ratios[i - 1],
            l1_ratios[i]
        );
    }
}

#[test]
fn test_elastic_net_prediction() {
    let features = Matrix::from_vec(
        8,
        3,
        vec![
            1.0, 0.5, 0.2, 2.0, 1.0, 0.4, 3.0, 1.5, 0.6, 4.0, 2.0, 0.8, 5.0, 2.5, 1.0, 6.0, 3.0,
            1.2, 7.0, 3.5, 1.4, 8.0, 4.0, 1.6,
        ],
    );
    let targets = Vector::from_vec(vec![2.1, 4.2, 5.9, 8.1, 10.2, 11.8, 14.1, 15.9]);

    let model = elastic_net::fit(&features, &targets, 0.1, 0.5, 1000, 1e-6).unwrap();
    let predictions = base::predict(&model, &features).unwrap();

    let r2 = base::score(&predictions, &targets).unwrap();

    assert!(r2 > 0.90, "R² score should be high, got {}", r2);
}
