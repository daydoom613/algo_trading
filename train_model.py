import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. IMPORT DATA
# ---------------------------------------------------------
print("Importing data from data_processor.py...")
from data_processor import X, y

# ---------------------------------------------------------
# 2. DEFINE TCN MODEL ARCHITECTURE
# ---------------------------------------------------------
def build_tcn_model(input_shape):
    model = Sequential([
        # LAYER 1: Immediate perception (1-minute patterns)
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=2, dilation_rate=1, padding='causal', activation='relu'),
        
        # LAYER 2: Short-term memory (2-minute patterns)
        Conv1D(filters=64, kernel_size=2, dilation_rate=2, padding='causal', activation='relu'),
        
        # LAYER 3: Mid-term trends (4-minute patterns)
        Conv1D(filters=64, kernel_size=2, dilation_rate=4, padding='causal', activation='relu'),
        
        # LAYER 4: The "Deep" look (8-minute patterns)
        Conv1D(filters=64, kernel_size=2, dilation_rate=8, padding='causal', activation='relu'),

        # FLATTEN: Convert the 3D tensor to 2D for the decision layer
        Flatten(),
        
        # DECISION LAYER: The "Synapse"
        Dense(32, activation='relu'),
        
        # OUTPUT LAYER: Predicted Velocity (1 neuron)
        Dense(1, activation='linear') 
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# ---------------------------------------------------------
# 3. EXECUTE TRAINING (The Learning Phase)
# ---------------------------------------------------------
# We wrap this in a try block to catch errors, but we define 'model' outside 
# so the next block can see it.
model = None 

try:
    print("\n--- Building TCN Architecture ---")
    # input_shape is derived from the imported X data
    model = build_tcn_model(input_shape=(X.shape[1], X.shape[2]))
    model.summary()
    
    print("\n--- Starting Training ---")
    # Epochs=20: It will read the entire dataset 20 times
    history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    print("✅ Training Complete.")

    # Visualize the Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss (Memory)', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss (Real Test)', color='orange')
    plt.title('The Learning Curve: Did the Bot understand the Physics?')
    plt.xlabel('Epochs')
    plt.ylabel('Error (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

except Exception as e:
    print(f"Error in training: {e}")
    exit()

# ---------------------------------------------------------
# 4. SAFETY & SIGNAL (Conformal Prediction)
# ---------------------------------------------------------
try:
    print("\n--- Starting Conformal Calibration ---")
    
    # 1. GET PREDICTIONS
    all_predictions = model.predict(X, verbose=0)
    
    # 2. CALIBRATE (The "Safety Inspection")
    # We look at the most recent 20% of data (The Validation Set) to measure accuracy
    split_idx = int(len(X) * 0.8)
    
    y_true_cal = y[split_idx:]
    y_pred_cal = all_predictions[split_idx:].flatten()
    
    # Calculate Absolute Errors (How wrong were we?)
    scores = np.abs(y_true_cal - y_pred_cal)
    
    # 3. COMPUTE THE SAFETY BUFFER (q_hat)
    confidence_level = 0.90
    alpha = 1 - confidence_level
    q_hat = np.quantile(scores, 1 - alpha)
    
    print(f"✅ Calibration Complete.")
    print(f"Safety Buffer (q_hat): {q_hat:.5f}")
    
    # 4. GENERATE LIVE SIGNAL
    latest_pred = all_predictions[-1][0]
    
    lower_bound = latest_pred - q_hat
    upper_bound = latest_pred + q_hat
    
    print(f"\n" + "="*40)
    print(f"   RELIANCE INDUSTRIES: LIVE AI SIGNAL")
    print(f"="*40)
    print(f"Predicted Velocity: {latest_pred:.6f}")
    print(f"90% Confidence Interval: [{lower_bound:.6f}, {upper_bound:.6f}]")
    
    # 5. THE DECISION LOGIC
    if lower_bound > 0:
        print(f"🚀 SIGNAL: BUY (Long)")
        print(f"   Why: Even in the worst-case scenario, price goes UP.")
    elif upper_bound < 0:
        print(f"wv SIGNAL: SELL (Short)")
        print(f"   Why: Even in the worst-case scenario, price goes DOWN.")
    else:
        print(f"⏸ SIGNAL: HOLD (No Trade)")
        print(f"   Why: The uncertainty range overlaps Zero. Too risky.")
        
    print("="*40)

except Exception as e:
    print(f"Error in calibration: {e}")