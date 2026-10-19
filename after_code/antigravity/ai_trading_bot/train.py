import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import pickle

import config
import data_processor
import model_builder

def train():
    # 1. Data Acquisition and Processing
    full_data = data_processor.get_data(config.TICKER, config.START_TRAIN, config.END_TEST)
    processed_data = data_processor.add_technical_indicators(full_data)
    labeled_data = data_processor.create_labels(processed_data, horizon=config.PREDICT_HORIZON)

    # Split into Train (2015-2023) and Test (2024)
    # Note: simple date string string comparison works for YYYY-MM-DD
    train_df = labeled_data.loc[config.START_TRAIN:config.END_TRAIN].copy()
    test_df = labeled_data.loc[config.START_TEST:config.END_TEST].copy()

    print(f"Training Set Size: {len(train_df)}")
    print(f"Testing Set Size: {len(test_df)}")

    if len(train_df) == 0:
        raise ValueError("Training set is empty. Check date ranges.")

    # 2. Sequencing and Normalization
    feature_cols = ['Close', 'RSI', 'MACD', 'Signal_Line', 'PctB']
    
    # Normalization fit ONLY on training data
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols])
    
    # Save the scaler for later use in evaluation
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create Sequences
    # We don't necessarily need sequences for the test set during training, 
    # but we prepare train_sequences strictly.
    X_train, y_train = data_processor.create_sequences(train_features, train_df['Label'].values, config.SEQ_LEN)
    
    # One-Hot Encode Targets
    y_train_encoded = to_categorical(y_train, num_classes=3)

    # 3. Fault-Tolerant Training Loop
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model_epoch_{epoch:02d}.keras")
    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='loss', # Monitoring loss since we might not have a validation set split out from train_df explicitly here
        save_best_only=False, # Save all to allow exact resumption
        save_weights_only=False,
        verbose=1
    )

    initial_epoch = 0
    model = None

    # Scan for existing checkpoints
    existing_checkpoints = glob.glob(os.path.join(config.CHECKPOINT_DIR, "model_epoch_*.keras"))

    if existing_checkpoints:
        # Find the latest file based on epoch number in filename
        # Assumes format "model_epoch_XX.keras"
        try:
            latest_file = max(existing_checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
            print(f"Resuming training from checkpoint: {latest_file}")
            
            # Load model state
            model = load_model(latest_file)
            
            # Extract epoch number
            initial_epoch = int(latest_file.split('_')[-1].split('.')[0])
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            model = model_builder.build_lstm_model((config.SEQ_LEN, len(feature_cols)))
    else:
        print("No checkpoints found. Starting fresh training.")
        model = model_builder.build_lstm_model((config.SEQ_LEN, len(feature_cols)))

    # Execute Training
    # Using a validation split from the training data itself
    if model:
        try:
           history = model.fit(
                X_train, y_train_encoded,
                epochs=10,                  # Reduced to 10 for quick verification, usually 50+
                batch_size=64,
                validation_split=0.1,
                callbacks=[checkpoint_cb],
                initial_epoch=initial_epoch # Resume internal counters
            )
           # Save final model explicitly as well
           model.save("final_model.keras")
           print("Training complete. Final model saved.")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Checkpoints should be saved.")
        except Exception as e:
            print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    train()
