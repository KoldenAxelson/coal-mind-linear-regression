use coal_mind_linear_regression::base::{self, LinearRegressionModel};
use nalgebra as na;
use plotters::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

const NUM_POINTS: usize = 100;
const OUTPUT_FILE: &str = "examples/output/basic_chart.png";

/// Generates synthetic data with some noise for demonstration
fn generate_sample_data() -> (na::DMatrix<f64>, na::DVector<f64>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 0.5).unwrap();

    // Generate x values between 0 and 10
    let x_values: Vec<f64> = (0..NUM_POINTS)
        .map(|i| i as f64 * 10.0 / NUM_POINTS as f64)
        .collect();

    // True relationship: y = 2x + 1 + noise
    let y_values: Vec<f64> = x_values
        .iter()
        .map(|&x| 2.0 * x + 1.0 + normal.sample(&mut rng))
        .collect();

    // Create matrix and vector
    let features = na::DMatrix::from_vec(NUM_POINTS, 1, x_values);
    let targets = na::DVector::from_vec(y_values);

    (features, targets)
}

/// Creates a scatter plot with the regression line
fn plot_regression(
    features: &na::DMatrix<f64>,
    targets: &na::DVector<f64>,
    model: &LinearRegressionModel,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a new drawing area
    let root = BitMapBackend::new(OUTPUT_FILE, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find data ranges for plot boundaries
    let x_min = features.min();
    let x_max = features.max();
    let y_min = targets.min();
    let y_max = targets.max();

    // Create chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Linear Regression Example", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d((x_min - 0.5)..(x_max + 0.5), (y_min - 0.5)..(y_max + 0.5))?;

    // Draw grid and axis
    chart.configure_mesh().draw()?;

    // Plot scatter points
    chart
        .draw_series(
            features
                .column(0)
                .iter()
                .zip(targets.iter())
                .map(|(&x, &y)| Circle::new((x, y), 3, BLUE.filled())),
        )?
        .label("Data Points")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // Generate points for regression line
    let line_x = [x_min, x_max];
    let line_y = line_x
        .iter()
        .map(|&x| model.coefficients[0] * x + model.intercept)
        .collect::<Vec<_>>();

    // Plot regression line
    chart
        .draw_series(LineSeries::new(
            line_x.iter().zip(line_y.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Regression Line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    // Draw legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    println!("Plot has been saved as '{}'", OUTPUT_FILE);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate sample data
    let (features, targets) = generate_sample_data();

    // Fit the model
    let model = base::fit(&features, &targets).expect("Failed to fit linear regression model");

    // Make predictions
    let predictions = base::predict(&model, &features).expect("Failed to make predictions");

    // Calculate and print metrics
    let r2_score = base::score(&predictions, &targets).expect("Failed to calculate R² score");
    let rmse = base::rmse(&predictions, &targets).expect("Failed to calculate RMSE");

    println!("Model Results:");
    println!("Coefficient (slope): {:.4}", model.coefficients[0]);
    println!("Intercept: {:.4}", model.intercept);
    println!("R² Score: {:.4}", r2_score);
    println!("RMSE: {:.4}", rmse);

    // Create visualization
    plot_regression(&features, &targets, &model)?;

    Ok(())
}
