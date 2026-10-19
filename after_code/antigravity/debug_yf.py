import yfinance as yf
import pandas as pd

ticker = "AAPL"
start = "2024-01-01"
end = "2024-01-10"

print(f"Downloading data for {ticker}...")
df = yf.download(ticker, start=start, end=end, progress=False)

print("\n--- DataFrame Head ---")
print(df.head())

print("\n--- DataFrame Columns ---")
print(df.columns)

print("\n--- DataFrame Index ---")
print(df.index)

print("\n--- Check 'Close' access ---")
try:
    print(df['Close'].head())
except Exception as e:
    print(f"Error accessing 'Close': {e}")

if isinstance(df.columns, pd.MultiIndex):
    print("\nColumns are MultiIndex")
    print("Level 0:", df.columns.get_level_values(0))
    print("Level 1:", df.columns.get_level_values(1))
