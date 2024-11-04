# Coal Mind Linear Regression Project Structure

## Directory Structure
```
coal-mind-linear-regression/
├── Cargo.toml                   # Dependencies / Set-Up
├── src/
│   ├── lib.rs                   # Library
│   ├── base.rs                  # [X] Basic Linear Regression
|   ├── lasso.rs                 # [X] L1 Linear Regression
|   ├── ridge.rs                 # [X] L2 Linear Regression
|   └── elastic_net.rs           # [X] L1 & L2 Linear Regression
├── tests/
│   ├── basic_test.rs            # [X] Basic Linear Regression Integration Testing
|   ├── lasso_test.rs            # [X] L1 Linear Regression Integration Testing
|   ├── ridge_test.rs            # [X] L2 Linear Regression Integration Testing
|   └── elastic_net_test.rs      # [X] L1 & L2 Linear Regression Integration Testing
├── benches/               
│   └── benchmarks.rs            # [ ] Benchmark
├── examples/  
│   ├── basic_example.rs         # [ ] Basic Linear Regression Example
│   ├── lasso_example.rs         # [ ] L1 Linear Regression Example
│   ├── ridge_example.rs         # [ ] L2 Linear Regression Example
│   └── elastic_net_example.rs   # [ ] L1 & L2 Linear Regression Example
└── README.md             
```

## Dependencies
- `nalgebra`: Linear algebra operations
- `criterion`: Benchmarking (dev-dependency)
