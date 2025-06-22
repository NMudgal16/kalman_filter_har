import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
data = np.loadtxt("C:/Users/kalman p/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt")


# Moving average
def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

# Kalman filter
def kalman_filter(signal):
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
    return np.array(filtered)

# Load signal
def load_signal(file_path):
    return np.loadtxt(file_path)

from sklearn.metrics import mean_squared_error

def add_noise(signal, noise_level=0.2):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def apply_kalman(signal):
    return kalman_filter(signal)

for i in [0, 20, 100]:
    original_signal = data[i]
    noisy_signal = add_noise(original_signal)

    kalman = apply_kalman(noisy_signal)
    moving_avg = moving_average(noisy_signal)
    savgol = savgol(noisy_signal)

    # Evaluate RMSE
    rmse_kalman = np.sqrt(mean_squared_error(original_signal, kalman))
    rmse_avg = np.sqrt(mean_squared_error(original_signal, moving_avg))
    rmse_savgol = np.sqrt(mean_squared_error(original_signal, savgol))

    print(f"[{sensor_name} - Sample {i}]")
    print(f"  RMSE Kalman       : {rmse_kalman:.4f}")
    print(f"  RMSE Moving Avg   : {rmse_avg:.4f}")
    print(f"  RMSE Savitzky-Golay: {rmse_savgol:.4f}")

    plot_all_filters(noisy_signal, kalman, moving_avg, savgol, i, sensor_name)

# Plot all filters for comparison
def compare_filters(signal, signal_name, sample_idx):
    kalman = kalman_filter(signal)
    moving_avg = moving_average(signal)
    savgol = savgol_filter(signal, window_length=9, polyorder=2)

    plt.figure(figsize=(12, 6))
    plt.plot(signal, label="Original", alpha=0.5)
    plt.plot(kalman, label="Kalman", linewidth=2)
    plt.plot(moving_avg, label="Moving Avg", linestyle='--')
    plt.plot(savgol, label="Savitzky-Golay", linestyle='-.')
    plt.title(f"{signal_name} Sample #{sample_idx} - Filter Comparison")
    plt.xlabel("Time step")
    plt.ylabel("Sensor value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{signal_name}_sample_{sample_idx}.png")
    plt.show()

def main():
    sensors = {
        "body_acc_x": "UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt",
        "body_acc_y": "UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt",
        "body_gyro_x": "UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt"
    }

    for name, path in sensors.items():
        print(f"\nProcessing {name}...")
        data = load_signal(path)
        for i in [0, 10, 50]:  # Pick a few sample windows
            signal = data[i]
            compare_filters(signal, name, i)

if __name__ == "__main__":
    main()
