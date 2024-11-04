use coal_mind_linear_regression::base::{self, LinearRegressionModel};
use coal_mind_linear_regression::ridge;
use nalgebra as na;
use plotters::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

const NUM_POINTS: usize = 100;
const NUM_FEATURES: usize = 3; // We'll use 3 features to demonstrate coefficient shrinkage
const OUTPUT_FILE: &str = "examples/output/ridge_chart.png";

/// Generates synthetic data with correlated features to demonstrate ridge regression's handling
/// of multicollinearity
fn generate_sample_data() -> (na::DMatrix<f64>, na::DVector<f64>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 0.3).unwrap();

    // Generate feature values
    let mut features = Vec::with_capacity(NUM_POINTS * NUM_FEATURES);
    let mut x1_values = Vec::with_capacity(NUM_POINTS);

    for i in 0..NUM_POINTS {
        // First feature: base signal
        let x1 = i as f64 * 10.0 / NUM_POINTS as f64;
        x1_values.push(x1);
        features.push(x1);

        // Second feature: correlated with first + noise
        features.push(0.7 * x1 + normal.sample(&mut rng));

        // Third feature: also correlated but different relationship
        features.push(0.3 * x1 * x1 + normal.sample(&mut rng));
    }

    // True relationship: y = x₁ + 0.5x₂ + 0.3x₃ + noise
    let y_values: Vec<f64> = (0..NUM_POINTS)
        .map(|i| {
            features[i * NUM_FEATURES]
                + 0.5 * features[i * NUM_FEATURES + 1]
                + 0.3 * features[i * NUM_FEATURES + 2]
                + normal.sample(&mut rng)
        })
        .collect();

    let features_matrix = na::DMatrix::from_vec(NUM_POINTS, NUM_FEATURES, features);
    let targets = na::DVector::from_vec(y_values);

    (features_matrix, targets)
}

/// Creates plots comparing different Ridge regularization strengths
fn plot_ridge_comparison(
    features: &na::DMatrix<f64>,
    targets: &na::DVector<f64>,
    models: &[(f64, LinearRegressionModel)], // (alpha, model) pairs
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(OUTPUT_FILE, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Split the drawing area into two panels
    let (left_panel, right_panel) = root.split_horizontally(500);

    // Left panel: Scatter plot with regression lines for feature 1
    {
        let x1_values: Vec<f64> = features.column(0).iter().cloned().collect();
        let x_min = x1_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x1_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = targets.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = targets.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&left_panel)
            .caption("Ridge Regression: Feature 1 vs Target", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d((x_min - 0.5)..(x_max + 0.5), (y_min - 0.5)..(y_max + 0.5))?;

        chart
            .configure_mesh()
            .x_desc("Feature 1")
            .y_desc("Target")
            .draw()?;

        // Plot scatter points
        chart
            .draw_series(
                x1_values
                    .iter()
                    .zip(targets.iter())
                    .map(|(&x, &y)| Circle::new((x, y), 3, BLUE)),
            )?
            .label("Data Points")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        // Plot regression lines for different alphas
        let colors = [&RED, &GREEN, &MAGENTA];
        for ((alpha, model), &color) in models.iter().zip(colors.iter()) {
            let line_x = [x_min, x_max];
            let line_y = line_x
                .iter()
                .map(|&x| model.coefficients[0] * x + model.intercept)
                .collect::<Vec<_>>();

            let alpha_label = format!("α = {:.3}", alpha);
            chart
                .draw_series(LineSeries::new(
                    line_x.iter().zip(line_y.iter()).map(|(&x, &y)| (x, y)),
                    color,
                ))?
                .label(&alpha_label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    // Right panel: Coefficient values vs regularization strength
    {
        let mut chart = ChartBuilder::on(&right_panel)
            .caption("Ridge Coefficient Paths", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                -5.0f64..1.0, // log10(alpha) range
                -0.5f64..2.5, // coefficient range
            )?;

        chart
            .configure_mesh()
            .x_desc("log₁₀(α)")
            .y_desc("Coefficient Value")
            .draw()?;

        // Plot coefficient paths
        let colors = [&RED, &GREEN, &MAGENTA];

        for (feature_idx, &color) in (0..NUM_FEATURES).zip(colors.iter()) {
            let coef_path: Vec<(f64, f64)> = models
                .iter()
                .map(|(alpha, model)| (alpha.log10(), model.coefficients[feature_idx]))
                .collect();

            let feature_label = format!("Feature {}", feature_idx + 1);
            chart
                .draw_series(LineSeries::new(coef_path, color))?
                .label(&feature_label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    root.present()?;
    println!("Plot has been saved as '{}'", OUTPUT_FILE);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate sample data
    let (features, targets) = generate_sample_data();

    // Fit models with different alpha values
    let alpha_values = [0.0, 0.1, 1.0, 10.0, 100.0];
    let mut models = Vec::new();

    for &alpha in &alpha_values {
        let model = if alpha == 0.0 {
            base::fit(&features, &targets)
        } else {
            ridge::fit(&features, &targets, alpha, 1000, 1e-4)
        }
        .expect("Failed to fit model");

        models.push((alpha, model));
    }

    // Print results for each model
    for (alpha, model) in &models {
        println!("\nResults for α = {:.3}:", alpha);
        println!("Coefficients:");
        for (i, &coef) in model.coefficients.iter().enumerate() {
            println!("  Feature {}: {:.4}", i + 1, coef);
        }
        println!("Intercept: {:.4}", model.intercept);

        // Calculate and print metrics
        let predictions = base::predict(model, &features).expect("Failed to make predictions");
        let r2_score = base::score(&predictions, &targets).expect("Failed to calculate R² score");
        let rmse = base::rmse(&predictions, &targets).expect("Failed to calculate RMSE");

        println!("R² Score: {:.4}", r2_score);
        println!("RMSE: {:.4}", rmse);
    }

    // Create visualization
    plot_ridge_comparison(&features, &targets, &models)?;

    Ok(())
}
