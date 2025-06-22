import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import os

# Load .txt signal files
def load_signal(file_path):
    return np.loadtxt(file_path)

# Add synthetic noise
def add_noise(signal, noise_level=0.2):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

# Kalman Filter
def apply_kalman(signal):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.], [0.]])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 1000.
    kf.R = 0.1
    kf.Q = 0.01

    filtered = []
    for z in signal:
        kf.predict()
        kf.update(z)
        filtered.append(kf.x[0, 0])
    return np.array(filtered), kf

# Moving Average
def apply_moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')

# Savitzky-Golay Filter
def apply_savgol(signal, window=11, polyorder=2):
    return savgol_filter(signal, window_length=window, polyorder=polyorder)

# Forecast using Kalman Filter (future N steps)
def forecast_kalman(kf, steps=10):
    forecasts = []
    for _ in range(steps):
        kf.predict()
        forecasts.append(kf.x[0, 0])
    return forecasts

# Plot & compare filters
def plot_all_filters(original, noisy, kalman, ma, sg, sample_idx, sensor_name):
    plt.figure(figsize=(12, 6))
    plt.plot(noisy, label='Noisy', alpha=0.4)
    plt.plot(kalman, label='Kalman', linewidth=2)
    plt.plot(ma, label='Moving Avg', linestyle='--')
    plt.plot(sg, label='Savitzky-Golay', linestyle=':')
    plt.title(f"{sensor_name} - Sample #{sample_idx}")
    plt.xlabel("Time step")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{sensor_name}_sample_{sample_idx}.png")
    plt.close()

# Main logic
def main():
    print("Starting Kalman Filter Project on HAR Dataset...\n")

    # List of all 9 Inertial Signals in UCI HAR
    sensors = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    # Choose samples to test
    sample_ids = [0, 20, 100]

    for sensor in sensors:
        file_path = f"UCI HAR Dataset/train/Inertial Signals/{sensor}_train.txt"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Processing sensor: {sensor}")
        data = load_signal(file_path)

        for sample_idx in sample_ids:
            original = data[sample_idx]
            noisy = add_noise(original, noise_level=0.2)

            kalman_filtered, kf = apply_kalman(noisy)
            moving_avg = apply_moving_average(noisy)
            savgol = apply_savgol(noisy)

            # Forecast next 10 values
            forecast = forecast_kalman(kf, steps=10)

            # RMSE Evaluation
            rmse_kalman = np.sqrt(mean_squared_error(original, kalman_filtered))
            rmse_ma = np.sqrt(mean_squared_error(original, moving_avg))
            rmse_sg = np.sqrt(mean_squared_error(original, savgol))

            print(f" Sample #{sample_idx} RMSE:")
            print(f"   Kalman       : {rmse_kalman:.4f}")
            print(f"   Moving Avg   : {rmse_ma:.4f}")
            print(f"   Savitzky-Golay: {rmse_sg:.4f}")

            # Plot
            plot_all_filters(original, noisy, kalman_filtered, moving_avg, savgol, sample_idx, sensor)

            # Plot forecast (optional)
            plt.figure(figsize=(10, 4))
            plt.plot(kalman_filtered, label="Kalman Filtered")
            plt.plot(range(len(kalman_filtered), len(kalman_filtered)+len(forecast)), forecast, 'r--', label="Forecast")
            plt.title(f"{sensor} - Forecast after Sample #{sample_idx}")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"plots/{sensor}_forecast_sample_{sample_idx}.png")
            plt.close()

    print("\n✅ All signals processed. Plots saved in 'plots/' folder.")

if __name__ == "__main__":
    main()

