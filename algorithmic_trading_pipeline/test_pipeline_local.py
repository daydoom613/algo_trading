"""
Test pipeline with local simulated data (no network required)
"""
import numpy as np
from main_pipeline import GreyBoxTradingEngine

# Create synthetic price data
np.random.seed(42)
n_points = 550

# Simulate realistic price data with drift and mean reversion
prices = np.zeros(n_points)
prices[0] = 100

for i in range(1, n_points):
    # Add trend, mean reversion, and noise
    drift = 0.0001
    mean_reversion = 0.01 * (100 - prices[i-1])
    noise = np.random.normal(0, 0.5)
    prices[i] = prices[i-1] + drift + mean_reversion + noise

print(f"Generated synthetic price data: {n_points} points")
print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
print(f"Mean: {prices.mean():.2f}, Std: {prices.std():.2f}")

# Create engine
engine = GreyBoxTradingEngine(prices=prices, tcn_channels=16)

# Run pipeline
results = engine.run_full_pipeline()

print("\n" + "="*70)
print("PIPELINE RESULTS WITH SYNTHETIC DATA")
print("="*70)
print(f"Signal Processing: [OK]")
if 'tda_analysis' in results:
    tda = results['tda_analysis']
    regime_switches = tda.get('regime_switches', 0)
    if isinstance(regime_switches, list):
        regime_switches = len(regime_switches)
    print(f"TDA Detections: {regime_switches} regime switches")

if 'prediction' in results:
    print(f"Predictions Generated: {len(results['prediction'].get('predictions', []))}")

if 'backtesting' in results:
    backtest = results['backtesting']
    print(f"Backtest Performance:")
    print(f"  Total Return: {backtest.get('total_return', 0):.2%}")
    print(f"  Annual Return: {backtest.get('annual_return', 0):.2%}")
    print(f"  Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {backtest.get('max_drawdown', 0):.2%}")
    print(f"  Win Rate: {backtest.get('win_rate', 0):.2%}")
    print(f"  Number of Trades: {backtest.get('num_trades', 0)}")
