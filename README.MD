# Kalman Filter on HAR Dataset

## Goal
Apply Kalman Filter to denoise body acceleration data (`body_acc_x_train.txt`) from the UCI HAR Dataset.

## Dataset
- Source: UCI Human Activity Recognition Dataset
- Sensor: Smartphone accelerometer (50Hz)

## Steps
1. Load 1D time series from a sample window (128 readings)
2. Apply Kalman Filter
3. Compare original vs filtered signal
4. Save plots

## Libraries
- numpy
- matplotlib
- filterpy

## To Run
```bash
pip install -r requirements.txt
python kalman_har.py
