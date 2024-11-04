# Coal Mind Linear Regression

Part of the CoalMind machine learning library family, this crate provides efficient implementations of linear regression algorithms including:
- Basic Linear Regression (Ordinary Least Squares)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net Regression (Combined L1 and L2 regularization)

## Features

- Pure Rust implementation with minimal dependencies
- Built on `nalgebra` for efficient matrix operations
- Comprehensive benchmarking suite
- Statistical visualization examples
- Memory-efficient data handling
- Well-documented mathematical principles

## Implementations

### Basic Linear Regression
- Solves Xβ = y using normal equations
- Direct matrix solution for optimal coefficients
- Fast and efficient for well-conditioned problems

### Ridge Regression (L2)
- Adds α||β||² penalty term for coefficient shrinkage
- Improves stability for multicollinear features
- Closed-form solution via regularized normal equations

### Lasso Regression (L1)
- Adds α||β||₁ penalty for sparse solutions
- Performs automatic feature selection
- Implemented using coordinate descent optimization

### Elastic Net
- Combines L1 and L2 regularization
- Balances feature selection and stability
- Configurable ratio between L1 and L2 penalties

### Polynomial Regression ( In Progress )
- Extends linear regression by adding polynomial terms of features
- Captures non-linear relationships between features and target
- Achieved by transforming features into polynomial features before fitting

## Usage

Add this to your `Cargo.toml`:
```toml
[dependencies]
coal-mind-linear-regression = "0.1.0"
```

Basic example:
```rust
use coal_mind_linear_regression::base;
use nalgebra as na;

// Prepare your data
let features = na::DMatrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let targets = na::DVector::from_vec(vec![2.0, 4.0, 6.0]);

// Fit the model
let model = base::fit(&features, &targets).unwrap();

// Make predictions
let predictions = base::predict(&model, &features).unwrap();

// Calculate R² score
let r2 = base::score(&predictions, &targets).unwrap();
```

## Performance

Benchmark results show relative performance characteristics:
- Base Linear Regression: Fastest (4-9µs for small datasets)
- Ridge Regression: Similar to base (~10µs)
- Lasso Regression: Slower due to iteration (60-200µs)
- Elastic Net: Similar to Lasso (170-200µs)

Performance varies with dataset size and regularization parameters.

## Examples

The `examples/` directory contains:
- `basic_example.rs`: Simple linear regression with visualization
- `lasso_example.rs`: Feature selection demonstration
- `ridge_example.rs`: Handling multicollinearity
- `elastic_net_example.rs`: Combined regularization effects

## Part of CoalMind

This crate is one of 20 foundational machine learning algorithms implemented in the CoalMind library family. Each implementation focuses on:
- Clean, idiomatic Rust code
- Educational value through clear documentation
- Production-ready performance
- Comprehensive testing and benchmarking

## Dependencies

- `nalgebra`: Linear algebra operations
- `plotters`: (dev) Visualization for examples
- `criterion`: (dev) Benchmarking
- `rand` and `rand_distr`: (dev) Sample data generation

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
