import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_traffic_data():
    """Load the traffic dataset."""
    print("Loading traffic data...")
    mat_data = loadmat('traffic+flow+forecasting-1/traffic_dataset.mat')

    # Extract target data (transpose to get time_steps x sensors)
    Y_train = mat_data['tra_Y_tr'].T  # (1261, 36)
    Y_test = mat_data['tra_Y_te'].T   # (840, 36)

    print(f"Training shape: {Y_train.shape}")
    print(f"Test shape: {Y_test.shape}")

    return Y_train, Y_test

def create_time_series_features(data, window_size=5):
    """Create time series features using sliding window."""
    print(f"Creating time series features with window size {window_size}...")

    X, y = [], []

    for i in range(window_size, len(data)):
        # Use previous window_size time steps as features
        X.append(data[i-window_size:i].flatten())
        # Predict current time step
        y.append(data[i])

    return np.array(X), np.array(y)

def train_models(X_train, y_train, X_test, y_test):
    """Train and evaluate multiple regression models."""
    print("\nTraining models...")

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }

    results = {}

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        print(f"  Training {name}...")

        # Train model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': y_pred_test,
            'actual': y_test
        }

        print(f"    Test MSE: {test_mse:.6f}, Test R²: {test_r2:.4f}")

    return results, scaler

def save_predictions(results):
    """Save model predictions to files."""
    print("\nSaving predictions...")

    for name, result in results.items():
        # Save predictions
        pred_filename = f"{name.lower().replace(' ', '_')}_predictions.csv"
        pd.DataFrame(result['predictions']).to_csv(pred_filename, index=False)

        # Save actual values (only once)
        if name == list(results.keys())[0]:  # Save actual only for first model
            pd.DataFrame(result['actual']).to_csv('actual_test_values.csv', index=False)

        print(f"  Saved {pred_filename}")

    print("  Saved actual_test_values.csv")
    print("  All predictions saved successfully!")

def plot_results(results):
    """Plot model performance and predictions."""
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Performance comparison
    model_names = list(results.keys())
    test_mse = [results[name]['test_mse'] for name in model_names]
    test_r2 = [results[name]['test_r2'] for name in model_names]

    axes[0,0].bar(model_names, test_mse)
    axes[0,0].set_ylabel('Test MSE')
    axes[0,0].set_title('Model Performance - Mean Squared Error')
    axes[0,0].tick_params(axis='x', rotation=45)

    axes[0,1].bar(model_names, test_r2)
    axes[0,1].set_ylabel('Test R²')
    axes[0,1].set_title('Model Performance - R² Score')
    axes[0,1].tick_params(axis='x', rotation=45)

    # Get best model for detailed plots
    best_model = min(results.keys(), key=lambda k: results[k]['test_mse'])
    best_results = results[best_model]

    # Prediction vs Actual scatter plot (sample subset)
    sample_size = min(1000, len(best_results['actual'].flatten()))
    indices = np.random.choice(len(best_results['actual'].flatten()), sample_size, replace=False)

    actual_sample = best_results['actual'].flatten()[indices]
    pred_sample = best_results['predictions'].flatten()[indices]

    axes[1,0].scatter(actual_sample, pred_sample, alpha=0.5, s=10)
    axes[1,0].plot([actual_sample.min(), actual_sample.max()],
                   [actual_sample.min(), actual_sample.max()], 'r--', lw=2)
    axes[1,0].set_xlabel('Actual Traffic Flow')
    axes[1,0].set_ylabel('Predicted Traffic Flow')
    axes[1,0].set_title(f'Best Model: {best_model}\nPredictions vs Actual')

    # Time series plot for one sensor
    sensor_idx = 0
    time_steps = min(100, len(best_results['actual']))

    axes[1,1].plot(best_results['actual'][:time_steps, sensor_idx],
                   label='Actual', linewidth=2)
    axes[1,1].plot(best_results['predictions'][:time_steps, sensor_idx],
                   label='Predicted', linewidth=2, alpha=0.8)
    axes[1,1].set_xlabel('Time Steps')
    axes[1,1].set_ylabel('Traffic Flow')
    axes[1,1].set_title(f'Time Series - Sensor {sensor_idx+1}')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simple_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Results saved as 'simple_prediction_results.png'")

def main():
    """Main function to run traffic flow prediction."""
    print("=== SIMPLE TRAFFIC FLOW PREDICTION ===\n")

    # Load data
    Y_train, Y_test = load_traffic_data()

    # Create time series features
    window_size = 5
    X_train, y_train = create_time_series_features(Y_train, window_size)

    # For test set, use end of training data as context
    Y_combined = np.vstack([Y_train[-window_size:], Y_test])
    X_test, y_test = create_time_series_features(Y_combined, window_size)

    print(f"\nFeature shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Train models
    results, scaler = train_models(X_train, y_train, X_test, y_test)

    # Save predictions
    save_predictions(results)

    # Plot results
    plot_results(results)

    # Print summary
    print("\n=== RESULTS SUMMARY ===")
    best_model = min(results.keys(), key=lambda k: results[k]['test_mse'])
    best_mse = results[best_model]['test_mse']
    best_r2 = results[best_model]['test_r2']

    print(f"Best model: {best_model}")
    print(f"Test MSE: {best_mse:.6f}")
    print(f"Test R²: {best_r2:.4f}")
    print(f"RMSE: {np.sqrt(best_mse):.6f}")

    for name, result in results.items():
        print(f"{name:20} | MSE: {result['test_mse']:.6f} | R²: {result['test_r2']:.4f}")

    return results

if __name__ == "__main__":
    results = main()