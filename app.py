import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter

# Load 1 sample signal
data = np.loadtxt("UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt")
signal = data[0]

st.title("Kalman Filter vs Smoothing Techniques")
st.write("Compare Kalman, Moving Average, and Savitzky-Golay filters on noisy sensor data.")

noise_level = st.slider("Noise Level", 0.0, 1.0, 0.2, 0.05)
noisy_signal = signal + np.random.normal(0, noise_level, size=signal.shape)

# Kalman Filter
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[0.], [0.]])
kf.F = np.array([[1., 1.], [0., 1.]])
kf.H = np.array([[1., 0.]])
kf.P *= 1000.
kf.R = 0.1
kf.Q = 0.01
kalman = []
for z in noisy_signal:
    kf.predict()
    kf.update(z)
    kalman.append(kf.x[0, 0])

# Moving Average
ma = np.convolve(noisy_signal, np.ones(5)/5, mode='same')

# Savitzky-Golay
sg = savgol_filter(noisy_signal, 11, 2)

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(noisy_signal, label="Noisy", alpha=0.4)
ax.plot(kalman, label="Kalman", linewidth=2)
ax.plot(ma, label="Moving Avg", linestyle="--")
ax.plot(sg, label="Savitzky-Golay", linestyle=":")
ax.legend()
st.pyplot(fig)
