import os

# --- Configuration Constants ---
TICKER = "AAPL"          # Target Asset
START_TRAIN = "2015-01-01"
END_TRAIN = "2023-12-31"
START_TEST = "2024-01-01"
END_TEST = "2024-12-31"  # Or current date
SEQ_LEN = 60             # Lookback window (e.g., 60 days)
PREDICT_HORIZON = 1      # Predict label for 1 step ahead
CHECKPOINT_DIR = "./training_checkpoints"

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
