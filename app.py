import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import os

st.set_page_config(layout="wide")
st.title("🔍 Kalman Filter + Smoothing on UCI HAR Inertial Signals")
st.markdown("Compare Kalman Filter, Moving Average, and Savitzky-Golay filtering on real accelerometer/gyroscope data with added noise.")

# Sensor list
SENSORS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

# Functions
@st.cache_data
def load_signal(sensor):
    path = f"UCI HAR Dataset/train/Inertial Signals/{sensor}_train.txt"
    return np.loadtxt(path)

def add_noise(signal, noise_level=0.2):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

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

def forecast_kalman(kf, steps=10):
    forecasts = []
    for _ in range(steps):
        kf.predict()
        forecasts.append(kf.x[0, 0])
    return forecasts

def apply_moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')

def apply_savgol(signal, window=11, polyorder=2):
    return savgol_filter(signal, window_length=window, polyorder=polyorder)

# Sidebar controls
sensor = st.sidebar.selectbox("Select Inertial Signal", SENSORS)
sample_idx = st.sidebar.slider("Sample Index", 0, 7351, 0)
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.2, 0.05)

# Load and process
signal_data = load_signal(sensor)
original = signal_data[sample_idx]
noisy = add_noise(original, noise_level=noise_level)

# Apply filters
kalman_filtered, kf = apply_kalman(noisy)
moving_avg = apply_moving_average(noisy)
savgol = apply_savgol(noisy)

# Forecast
forecast = forecast_kalman(kf, steps=10)

# RMSE
rmse_kalman = np.sqrt(mean_squared_error(original, kalman_filtered))
rmse_ma = np.sqrt(mean_squared_error(original, moving_avg))
rmse_sg = np.sqrt(mean_squared_error(original, savgol))

# 📊 Plot filters
st.subheader("📈 Filter Comparison on Noisy Signal")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(noisy, label='Noisy', alpha=0.3)
ax1.plot(kalman_filtered, label='Kalman Filter', linewidth=2)
ax1.plot(moving_avg, label='Moving Average', linestyle='--')
ax1.plot(savgol, label='Savitzky-Golay', linestyle=':')
ax1.set_title(f"{sensor} — Sample #{sample_idx}")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# 📈 Plot forecast
st.subheader("🔮 Forecast (Next 10 Values) using Kalman Filter")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(kalman_filtered, label="Kalman Filtered")
ax2.plot(range(len(kalman_filtered), len(kalman_filtered) + 10), forecast, 'r--', label="Forecast")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# 📉 RMSE Scores
st.subheader("📊 RMSE Comparison")
st.markdown(f"""
- **Kalman Filter RMSE**: `{rmse_kalman:.4f}`  
- **Moving Average RMSE**: `{rmse_ma:.4f}`  
- **Savitzky-Golay RMSE**: `{rmse_sg:.4f}`
""")
