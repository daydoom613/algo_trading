import yfinance as yf
import pywt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 1. FETCH DATA
# ---------------------------------------------------------
ticker = "RELIANCE.NS"
print(f"Fetching data for {ticker}...")
data = yf.download(ticker, period="59d", interval="5m", progress=False, auto_adjust=True)

if data.empty:
    print("❌ Error: No data fetched.")
    exit()

prices = data['Close'].values.flatten()
print(f"✅ Downloaded {len(prices)} candles.")

# ---------------------------------------------------------
# 2. WAVELET DENOISING
# ---------------------------------------------------------
def apply_wavelet_denoising(signal, wavelet='sym5', level=2):
    # Handle Divisibility
    required_divisibility = 2 ** level
    remainder = len(signal) % required_divisibility
    if remainder != 0:
        signal = signal[remainder:]
    
    # Right-Padding
    pad_length = 32 
    padded_signal = np.pad(signal, (0, pad_length), mode='edge')
    
    current_len = len(padded_signal)
    rem_pad = current_len % required_divisibility
    if rem_pad != 0:
        extra_needed = required_divisibility - rem_pad
        padded_signal = np.pad(padded_signal, (0, extra_needed), mode='edge')

    # Decompose & Denoise
    coeffs = pywt.swt(padded_signal, wavelet, level=level)
    denoised_coeffs = []
    for cA, cD in coeffs:
        zeros = np.zeros_like(cD) 
        denoised_coeffs.append((cA, zeros))
    
    clean_padded = pywt.iswt(denoised_coeffs, wavelet)
    clean_signal = clean_padded[:len(signal)]
    return clean_signal

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------
def prepare_tcn_data(prices, window_size=60, forecast_horizon=1):
    p = np.array(prices)
    
    # Physics Features
    velocity = np.diff(np.log(p))
    velocity = np.insert(velocity, 0, 0)
    
    acceleration = np.diff(velocity)
    acceleration = np.insert(acceleration, 0, 0)
    
    energy = pd.Series(velocity).rolling(window=10).std().fillna(0).values

    features = np.stack([velocity, acceleration, energy], axis=1)

    # Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_features = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(window_size, len(scaled_features) - forecast_horizon):
        X.append(scaled_features[i-window_size:i])
        y.append(scaled_features[i+forecast_horizon, 0]) 
        
    return np.array(X), np.array(y), scaler

# ---------------------------------------------------------
# EXECUTION (Calculates X and y)
# ---------------------------------------------------------
print("\n--- Processing Data in background ---")
denoised_prices = apply_wavelet_denoising(prices, level=2)
WINDOW_SIZE = 60 
X, y, scaler = prepare_tcn_data(denoised_prices, window_size=WINDOW_SIZE)
print(f"✅ Data Ready: X shape {X.shape}, y shape {y.shape}")