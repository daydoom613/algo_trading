import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.utils import to_categorical
import pickle
import os

import config
import data_processor

def evaluate():
    # 1. Load Model and Scaler
    model_path = "final_model.keras"
    if not os.path.exists(model_path):
        # Try finding latest checkpoint if final model not present
        import glob
        existing_checkpoints = glob.glob(os.path.join(config.CHECKPOINT_DIR, "model_epoch_*.keras"))
        if existing_checkpoints:
             model_path = max(existing_checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        else:
            print("No model found to evaluate.")
            return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    if not os.path.exists('scaler.pkl'):
        print("Scaler not found. Run training first.")
        return
        
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # 2. Data Acquisition (Test Set)
    # We fetch full data strategies to calculate indicators correctly for the start of 2024
    # Fetching a bit of 2023 to ensure indicators are primed
    full_data = data_processor.get_data(config.TICKER, "2023-11-01", config.END_TEST) 
    processed_data = data_processor.add_technical_indicators(full_data)
    labeled_data = data_processor.create_labels(processed_data, horizon=config.PREDICT_HORIZON)
    
    test_df = labeled_data.loc[config.START_TEST:config.END_TEST].copy()
    print(f"Testing Set Size: {len(test_df)}")
    
    feature_cols = ['Close', 'RSI', 'MACD', 'Signal_Line', 'PctB']
    test_features = scaler.transform(test_df[feature_cols])
    
    X_test, y_test = data_processor.create_sequences(test_features, test_df['Label'].values, config.SEQ_LEN)
    y_test_encoded = to_categorical(y_test, num_classes=3)

    # 3. Inference
    print("Generating predictions for 2024 data...")
    predictions_prob = model.predict(X_test)
    predictions_classes = np.argmax(predictions_prob, axis=1)
    true_classes = np.argmax(y_test_encoded, axis=1)

    # 4. Metrics
    unique, counts = np.unique(predictions_classes, return_counts=True)
    signal_counts = dict(zip(unique, counts))
    
    label_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    readable_counts = {label_map.get(k, k): v for k, v in signal_counts.items()}
    
    print("\n--- 2024 Signal Generation Report ---")
    print(f"Total Trading Opportunities Analyzed: {len(predictions_classes)}")
    print(f"Signal Distribution: {readable_counts}")

    acc = accuracy_score(true_classes, predictions_classes)
    conf_matrix = confusion_matrix(true_classes, predictions_classes)
    class_report = classification_report(true_classes, predictions_classes, target_names=['Hold', 'Buy', 'Sell'], zero_division=0)

    print(f"\nOverall Model Accuracy on 2024 Data: {acc:.2%}")
    print("\n--- Detailed Classification Report ---")
    print(class_report)
    
    print("\n--- Confusion Matrix ---")
    # Rows = True Classes, Columns = Predicted Classes
    print(pd.DataFrame(conf_matrix, 
                       index=['True Hold', 'True Buy', 'True Sell'], 
                       columns=['Pred Hold', 'Pred Buy', 'Pred Sell']))

if __name__ == "__main__":
    evaluate()
