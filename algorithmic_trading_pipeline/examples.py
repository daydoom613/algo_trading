"""
Quick Start Example: Running the Complete Algorithmic Trading Pipeline
Execute this script to test all components end-to-end
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_processing.modwt import CausalMODWT, WaveletVarianceAnalyzer
from tda.persistent_homology import MarketPhaseDetector
from prediction.tcn import TCNPredictor
from risk_management.conformal_prediction import RiskManager
from backtesting.vectorized_backtest import VectorizedBacktest, BacktestConfig
from utils.data_utils import DataPreprocessor, RealWorldDataContext, MultiAssetDataProcessor


def example_1_signal_processing():
    """Example 1: MODWT Signal Processing"""
    print("\n" + "="*70)
    print("EXAMPLE 1: MODWT Signal Processing")
    print("="*70)
    
    # Generate synthetic price series
    np.random.seed(42)
    trend = 100 + 0.05 * np.arange(500)
    noise = np.random.normal(0, 2, 500)
    prices = trend + noise
    
    # Initialize MODWT processor
    modwt = CausalMODWT(wavelet='sym8', decomposition_level=4)
    
    # Denoise signal
    denoised = modwt.wavelet_denoise(prices, threshold_multiplier=1.5)
    
    # Analyze volatility
    analyzer = WaveletVarianceAnalyzer()
    vol_analysis = analyzer.analyze_volatility_drivers(prices)
    
    print(f"Original series: mean={np.mean(prices):.2f}, std={np.std(prices):.2f}")
    print(f"Denoised series: mean={np.mean(denoised):.2f}, std={np.std(denoised):.2f}")
    print(f"\nVolatility Analysis:")
    print(f"  Short-term variance: {vol_analysis['short_term_variance_pct']:.1f}%")
    print(f"  Long-term variance: {vol_analysis['long_term_variance_pct']:.1f}%")
    print(f"  Noise dominant: {vol_analysis['noise_dominant']}")
    
    return denoised


def example_2_tda_analysis(prices):
    """Example 2: TDA Phase Detection"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Topological Data Analysis")
    print("="*70)
    
    detector = MarketPhaseDetector(embedding_dim=5, time_delay=1)
    
    crash_warnings = 0
    regime_switches = 0
    
    # Analyze price windows
    for i in range(100, len(prices) - 100, 50):
        window = prices[i-100:i+100]
        analysis = detector.analyze_window(window)
        
        if analysis['crash_warning']:
            crash_warnings += 1
            print(f"  ⚠️  Crash warning at index {i}, Z-score: {analysis['crash_zscore']:.2f}")
        
        if analysis['regime_switch']:
            regime_switches += 1
            print(f"  🔄 Regime switch at index {i}, Wasserstein: {analysis['wasserstein_distance']:.4f}")
    
    print(f"\nSummary:")
    print(f"  Crash warnings detected: {crash_warnings}")
    print(f"  Regime switches detected: {regime_switches}")


def example_3_prediction(prices):
    """Example 3: TCN Prediction"""
    print("\n" + "="*70)
    print("EXAMPLE 3: TCN Prediction")
    print("="*70)
    
    device = 'cpu'  # Use CPU for this example
    
    # Create simple features (price history)
    # Shape: (batch_size, sequence_length, input_channels)
    X = np.zeros((len(prices) - 10, 10, 1))
    y = np.zeros((len(prices) - 10, 10, 1))
    
    for i in range(len(prices) - 10):
        X[i, :, 0] = prices[i:i+10]
        y[i, :, 0] = prices[i+1:i+11]
    
    # Initialize predictor
    predictor = TCNPredictor(
        input_size=1, output_size=1,
        num_channels=16, kernel_size=3,
        num_levels=3, use_attention=True,
        device=device
    )
    
    print(f"TCN Configuration:")
    print(f"  Receptive field: {predictor.model.get_receptive_field()} steps")
    print(f"  Device: {device}")
    
    # Train for few epochs
    print(f"\nTraining...")
    for epoch in range(3):
        loss = predictor.train_step(X, y)
        print(f"  Epoch {epoch + 1}/3, Loss: {loss:.6f}")
    
    # Make predictions
    predictions = predictor.predict(X[:10])
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0, :5, 0]}")


def example_4_risk_management():
    """Example 4: Conformal Prediction & Risk Management"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Conformal Prediction & Position Sizing")
    print("="*70)
    
    rm = RiskManager(alpha=0.05, num_ensembles=10, base_position_size=1.0)
    
    # Simulate calibration period
    np.random.seed(42)
    for i in range(50):
        point_pred = 100 + i * 0.1
        residual = 1.0 + 0.3 * np.sin(i / 10)
        rm.enbpi.update_calibration(residual)
    
    # Generate predictions with intervals
    print("\nSample Predictions with Intervals:")
    print("-" * 70)
    print(f"{'Time':<8} {'Prediction':<12} {'Lower':<12} {'Upper':<12} {'Size':<8} {'Coverage':<12}")
    print("-" * 70)
    
    for i in range(5):
        point_pred = 105 + i * 0.5
        result = rm.predict_with_sizing(point_pred, estimated_residual=0.8)
        
        print(f"T+{i+1:<6} {result['point_prediction']:<12.2f} "
              f"{result['lower_bound']:<12.2f} {result['upper_bound']:<12.2f} "
              f"{result['position_size']:<8.2f} {result['target_coverage']:<12.1%}")
    
    print("-" * 70)
    
    metrics = rm.get_risk_metrics()
    print(f"\nRisk Metrics:")
    print(f"  Current alpha: {metrics['current_alpha']:.4f}")
    print(f"  Target coverage: {metrics['target_coverage']:.1%}")


def example_5_backtesting(prices):
    """Example 5: Vectorized Backtesting"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Vectorized Backtesting")
    print("="*70)
    
    # Simple signal: buy when price > MA20, sell otherwise
    def simple_signal(price_series):
        ma20 = np.convolve(price_series, np.ones(20)/20, mode='same')
        signals = np.where(price_series > ma20, 1, -1)
        signals[np.isnan(signals)] = 0
        return signals
    
    config = BacktestConfig(initial_capital=100000, commission=0.001)
    backtest = VectorizedBacktest(prices, config)
    
    signals = backtest.generate_signals(simple_signal)
    result = backtest.backtest_strategy(signals)
    
    print(f"\nBacktest Results:")
    print(f"  Total Return: {result['total_return']:.2%}")
    print(f"  Annual Return: {result['annual_return']:.2%}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"  Win Rate: {result['win_rate']:.2%}")
    print(f"  Number of Trades: {result['num_trades']}")


def example_6_real_world_data():
    """Example 6: Real-World Data - Reliance Industries"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Real-World Data Context - Reliance Industries")
    print("="*70)
    
    print("Fetching market data...")
    context = RealWorldDataContext.get_ril_context()
    
    print(f"Available assets: {list(context.keys())}")
    
    # Align data
    aligned = MultiAssetDataProcessor.align_data(context)
    
    print(f"\nAligned prices:")
    for asset, prices in aligned.items():
        print(f"  {asset:<12} {len(prices)} data points, "
              f"price range [{prices.min():.2f}, {prices.max():.2f}]")
    
    # Create multivariate features
    X, feature_names = MultiAssetDataProcessor.create_multivariate_features(
        aligned, n_lags=5
    )
    
    print(f"\nMultivariate features:")
    print(f"  Shape: {X.shape}")
    print(f"  Features: {feature_names[:8]}...")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("ALGORITHMIC TRADING PIPELINE - QUICK START EXAMPLES")
    print("="*70)
    
    # Example 1: Signal Processing
    denoised_prices = example_1_signal_processing()
    
    # Example 2: TDA
    example_2_tda_analysis(denoised_prices)
    
    # Example 3: Prediction
    example_3_prediction(denoised_prices)
    
    # Example 4: Risk Management
    example_4_risk_management()
    
    # Example 5: Backtesting
    example_5_backtesting(denoised_prices)
    
    # Example 6: Real-World Data
    try:
        example_6_real_world_data()
    except Exception as e:
        print(f"⚠️  Could not fetch real-world data: {e}")
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run full pipeline: python main_pipeline.py")
    print("  2. Start dashboard: streamlit run dashboard/app.py")
    print("  3. Review README.md for detailed documentation")


if __name__ == '__main__':
    main()
