use coal_mind_linear_regression::{base, elastic_net, lasso, ridge};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra as na;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// Generates synthetic data for benchmarking
fn generate_benchmark_data(
    n_samples: usize,
    n_features: usize,
) -> (na::DMatrix<f64>, na::DVector<f64>) {
    // Check for potential overflow in allocation size
    let total_elements = n_samples
        .checked_mul(n_features)
        .expect("Data size too large - would overflow");

    // Ensure reasonable size limits
    assert!(
        total_elements <= 100_000,
        "Data size too large - exceeding 100k elements"
    );

    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate feature matrix
    let features: Vec<f64> = (0..total_elements)
        .map(|_| normal.sample(&mut rng))
        .collect();

    // Generate true coefficients with decreasing importance
    let true_coefficients: Vec<f64> = (0..n_features).map(|i| 1.0 / (i + 1) as f64).collect();

    let features_matrix = na::DMatrix::from_vec(n_samples, n_features, features);
    let coef_vector = na::DVector::from_vec(true_coefficients);
    let noise: Vec<f64> = (0..n_samples).map(|_| normal.sample(&mut rng)).collect();
    let targets = &features_matrix * &coef_vector + na::DVector::from_vec(noise);

    (features_matrix, targets)
}

fn benchmark_base_regression(c: &mut Criterion) {
    let sizes = vec![(100, 5), (200, 10)];
    let mut group = c.benchmark_group("base_regression");

    for (n_samples, n_features) in sizes {
        let (features, targets) = generate_benchmark_data(n_samples, n_features);
        group.bench_with_input(
            BenchmarkId::new("base", format!("{}x{}", n_samples, n_features)),
            &(&features, &targets),
            |b, (x, y)| {
                b.iter(|| {
                    black_box(base::fit(x, y).unwrap());
                });
            },
        );
    }
    group.finish();
}

fn benchmark_ridge_regression(c: &mut Criterion) {
    let (features, targets) = generate_benchmark_data(200, 10);
    let mut group = c.benchmark_group("ridge_regression");

    for alpha in [0.1, 1.0] {
        group.bench_with_input(
            BenchmarkId::new("ridge", alpha),
            &(&features, &targets),
            |b, (x, y)| {
                b.iter(|| {
                    black_box(ridge::fit(x, y, alpha, 100, 1e-6).unwrap());
                });
            },
        );
    }
    group.finish();
}

fn benchmark_lasso_regression(c: &mut Criterion) {
    let (features, targets) = generate_benchmark_data(200, 10);
    let mut group = c.benchmark_group("lasso_regression");

    for alpha in [0.1, 1.0] {
        group.bench_with_input(
            BenchmarkId::new("lasso", alpha),
            &(&features, &targets),
            |b, (x, y)| {
                b.iter(|| {
                    black_box(lasso::fit(x, y, alpha, 100, 1e-6).unwrap());
                });
            },
        );
    }
    group.finish();
}

fn benchmark_elastic_net_regression(c: &mut Criterion) {
    let (features, targets) = generate_benchmark_data(200, 10);
    let mut group = c.benchmark_group("elastic_net_regression");

    for l1_ratio in [0.0, 0.5, 1.0] {
        group.bench_with_input(
            BenchmarkId::new("elastic_net", l1_ratio),
            &(&features, &targets),
            |b, (x, y)| {
                b.iter(|| {
                    black_box(elastic_net::fit(x, y, 0.1, l1_ratio, 100, 1e-6).unwrap());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_base_regression,
    benchmark_ridge_regression,
    benchmark_lasso_regression,
    benchmark_elastic_net_regression
);
criterion_main!(benches);
