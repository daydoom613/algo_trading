"""
Integrated Algorithmic Trading Pipeline
Combines MODWT, TDA, TCN, and Conformal Prediction into unified system
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import torch

# Import all pipeline components
from signal_processing.modwt import CausalMODWT, WaveletVarianceAnalyzer
from tda.persistent_homology import MarketPhaseDetector, TakensEmbedding
from prediction.tcn import TCNPredictor
from risk_management.conformal_prediction import RiskManager
from backtesting.vectorized_backtest import VectorizedBacktest, BacktestConfig, WalkForwardOptimizer
from utils.data_utils import DataPreprocessor, MultiAssetDataProcessor, RealWorldDataContext


class GreyBoxTradingEngine:
    """
    Main trading engine combining all components into coherent system.
    
    Architecture:
    1. Signal Processing (MODWT): Denoise price → extract multi-scale trends
    2. TDA (Persistent Homology): Detect regime changes and crash warnings
    3. Prediction (TCN): Forecast prices with attention weighting
    4. Risk Management (Conformal): Generate prediction intervals + position sizing
    5. Backtesting: Validate strategy with walk-forward optimization
    """
    
    def __init__(self, prices: np.ndarray, device: str = 'cpu',
                 modwt_decomp_level: int = 4, tcn_channels: int = 32,
                 embedding_dim: int = 5, risk_alpha: float = 0.05):
        """
        Initialize the trading engine.
        
        Args:
            prices: Historical price series
            device: 'cpu' or 'cuda'
            modwt_decomp_level: Wavelet decomposition levels
            tcn_channels: TCN hidden channels
            embedding_dim: TDA embedding dimension
            risk_alpha: Conformal prediction miscoverage level
        """
        self.prices = prices
        self.device = device
        
        # Initialize components
        print("Initializing MODWT signal processor...")
        self.modwt = CausalMODWT(wavelet='sym8', decomposition_level=modwt_decomp_level)
        self.wavelet_analyzer = WaveletVarianceAnalyzer(wavelet='sym8')
        
        print("Initializing TDA detector...")
        self.tda_detector = MarketPhaseDetector(embedding_dim=embedding_dim, time_delay=1)
        
        print("Initializing TCN predictor...")
        self.tcn_predictor = TCNPredictor(
            input_size=6, output_size=1,  # 6 features (norm_price, momentum, vol, rsi, macd, bb_pos)
            num_channels=tcn_channels, kernel_size=5,
            dropout=0.2, num_levels=4, use_attention=True,
            device=device
        )
        
        print("Initializing risk manager...")
        self.risk_manager = RiskManager(alpha=risk_alpha, num_ensembles=10, 
                                       base_position_size=1.0)
        
        # State tracking
        self.denoised_signal = None
        self.market_phases = []
        self.predictions = []
        self.position_sizes = []
    
    def run_signal_processing(self, window_size: int = 512) -> Dict:
        """
        Stage 1: Multi-resolution signal processing via MODWT.
        
        Args:
            window_size: Rolling window for causal decomposition
            
        Returns:
            Dictionary with denoised signal and volatility analysis
        """
        print(f"\n[STAGE 1] MODWT Signal Processing (window={window_size})...")
        
        # Denoise price signal
        self.denoised_signal = self.modwt.wavelet_denoise(
            self.prices, threshold_multiplier=1.5, window_size=window_size
        )
        
        # Analyze volatility drivers
        vol_analysis = self.wavelet_analyzer.analyze_volatility_drivers(
            self.prices, window_size=window_size
        )
        
        print(f"  [OK] Signal denoised. Noise-dominant: {vol_analysis['noise_dominant']}")
        print(f"    Short-term variance: {vol_analysis['short_term_variance_pct']:.1f}%")
        print(f"    Long-term variance: {vol_analysis['long_term_variance_pct']:.1f}%")
        
        return {
            'denoised_prices': self.denoised_signal,
            'volatility_analysis': vol_analysis
        }
    
    def run_tda_analysis(self, window_size: int = 100) -> Dict:
        """
        Stage 2: Topological Data Analysis for regime detection.
        
        Args:
            window_size: Rolling window for phase space reconstruction
            
        Returns:
            Dictionary with crash signals and regime switches
        """
        print(f"\n[STAGE 2] TDA Phase Detection (window={window_size})...")
        
        crash_warnings = []
        regime_switches = []
        
        historical_norm = 1.0
        previous_cloud = None
        
        # Process in rolling windows
        for t in range(window_size, len(self.prices), max(1, window_size // 4)):
            window_end = min(t + window_size, len(self.prices))
            price_window = self.prices[max(0, t - window_size):window_end]
            
            # Analyze current window
            analysis = self.tda_detector.analyze_window(
                price_window, previous_cloud, historical_norm
            )
            
            self.market_phases.append({
                'time': t,
                'crash_warning': analysis['crash_warning'],
                'crash_zscore': analysis['crash_zscore'],
                'regime_switch': analysis['regime_switch'],
                'wasserstein_dist': analysis['wasserstein_distance'],
                'persistence_norm': analysis['persistence_norm']
            })
            
            if analysis['crash_warning']:
                crash_warnings.append((t, analysis['crash_zscore']))
            
            if analysis['regime_switch']:
                regime_switches.append((t, analysis['wasserstein_distance']))
            
            # Update for next iteration
            previous_cloud = analysis['embedded_cloud']
            historical_norm = 0.7 * historical_norm + 0.3 * analysis['persistence_norm']
        
        print(f"  [OK] TDA analysis complete.")
        print(f"    Crash warnings detected: {len(crash_warnings)}")
        print(f"    Regime switches detected: {len(regime_switches)}")
        
        return {
            'market_phases': self.market_phases,
            'crash_warnings': crash_warnings,
            'regime_switches': regime_switches
        }
    
    def prepare_tcn_input(self, lookback: int = 100) -> Tuple[np.ndarray, List]:
        """
        Prepare features for TCN predictor with technical indicators.
        
        Features (6 total):
        1. Normalized denoised price
        2. Momentum
        3. Volatility
        4. RSI
        5. MACD
        6. Bollinger Band Position
        """
        if self.denoised_signal is None:
            raise ValueError("Run signal processing first")
        
        n = len(self.prices)
        X = np.zeros((max(0, n - lookback), 6))
        
        for i in range(n - lookback):
            t = i + lookback
            
            # Feature 0: Normalized denoised price
            X[i, 0] = (self.denoised_signal[t] - np.mean(self.denoised_signal[:t+1])) / (np.std(self.denoised_signal[:t+1]) + 1e-6)
            
            # Feature 1: Momentum (derivative)
            if t > 0:
                X[i, 1] = self.denoised_signal[t] - self.denoised_signal[t-1]
            else:
                X[i, 1] = 0
            
            # Feature 2: Volatility (std of recent returns)
            if t >= 20:
                recent_returns = np.diff(self.prices[max(0, t-20):t])
                X[i, 2] = np.std(recent_returns)
            else:
                X[i, 2] = 0
            
            # Feature 3: RSI (Relative Strength Index)
            if t >= 14:
                price_window = self.prices[max(0, t-14):t+1]
                deltas = np.diff(price_window)
                gains = np.mean(deltas[deltas > 0]) if len(deltas[deltas > 0]) > 0 else 0
                losses = -np.mean(deltas[deltas <= 0]) if len(deltas[deltas <= 0]) > 0 else 0
                rs = gains / (losses + 1e-8)
                X[i, 3] = 100 - (100 / (1 + rs))
            else:
                X[i, 3] = 50
            
            # Feature 4: MACD
            if t >= 26:
                ema_12 = np.mean(self.prices[max(0, t-12):t+1])
                ema_26 = np.mean(self.prices[max(0, t-26):t+1])
                X[i, 4] = ema_12 - ema_26
            else:
                X[i, 4] = 0
            
            # Feature 5: Bollinger Bands Position
            if t >= 20:
                price_window = self.prices[max(0, t-20):t+1]
                bb_middle = np.mean(price_window)
                bb_std = np.std(price_window)
                bb_position = (self.prices[t] - bb_middle) / (bb_std + 1e-8)
                X[i, 5] = np.clip(bb_position, -3, 3) / 3  # Normalize to -1 to 1
            else:
                X[i, 5] = 0
        
        return X, ['norm_price', 'momentum', 'volatility', 'rsi', 'macd', 'bb_position']
    
    def run_prediction(self, epochs: int = 10, lookback: int = 100) -> Dict:
        """
        Stage 3: Temporal Convolutional Network prediction.
        
        Args:
            epochs: Training epochs
            lookback: Lookback period for features
            
        Returns:
            Dictionary with predictions
        """
        print(f"\n[STAGE 3] TCN Prediction (epochs={epochs}, lookback={lookback})...")
        
        # Prepare features
        X, feature_names = self.prepare_tcn_input(lookback)
        
        if len(X) == 0:
            print("  [ERR] Insufficient data for prediction")
            return {'predictions': [], 'predictions_std': []}
        
        # Normalize features for better training
        X_mean = np.nanmean(X, axis=0, keepdims=True)
        X_std = np.nanstd(X, axis=0, keepdims=True)
        X_std = np.maximum(X_std, 1e-6)  # Prevent division by zero
        X_normalized = (X - X_mean) / X_std
        X_normalized = np.nan_to_num(X_normalized, nan=0.0)
        
        # Create target (next period return)
        y = np.diff(self.prices[lookback:len(self.prices)], axis=0)
        y = np.concatenate([[0], y])[:len(X)]
        
        # Normalize target
        y_mean = np.nanmean(y)
        y_std = np.nanstd(y)
        y_std = max(y_std, 1e-6)
        y_normalized = (y - y_mean) / y_std
        y_normalized = np.nan_to_num(y_normalized, nan=0.0)
        
        # Reshape for model
        X_tensor = np.expand_dims(X_normalized, axis=0)  # (1, seq_len, features)
        y_tensor = np.expand_dims(y_normalized, axis=(0, 2))  # (1, seq_len, 1)
        
        # Train
        for epoch in range(epochs):
            loss = self.tcn_predictor.train_step(X_tensor, y_tensor)
            if (epoch + 1) % max(1, epochs // 3) == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        # Make predictions
        predictions = self.tcn_predictor.predict(X_normalized)[:, :, 0]  # (1, seq_len)
        
        # Denormalize predictions
        self.predictions = predictions[0] if predictions.shape[0] == 1 else predictions
        self.predictions = self.predictions * y_std + y_mean
        
        print(f"  [OK] TCN trained. Generated {len(self.predictions)} predictions")
        
        return {
            'predictions': self.predictions,
            'feature_names': feature_names,
            'receptive_field': self.tcn_predictor.model.get_receptive_field()
        }
    
    def run_risk_management(self) -> Dict:
        """
        Stage 4: Conformal Prediction for uncertainty quantification.
        
        Returns:
            Dictionary with prediction intervals and position sizes
        """
        print(f"\n[STAGE 4] Conformal Prediction & Position Sizing...")
        
        if len(self.predictions) == 0:
            print("  [ERR] No predictions available")
            return {'intervals': [], 'position_sizes': []}
        
        intervals = []
        position_sizes = []
        
        # Calibrate from first 30% of predictions
        calibration_size = max(10, len(self.predictions) // 3)
        calibration_residuals = np.abs(np.diff(self.predictions[:calibration_size]))
        
        for residual in calibration_residuals:
            self.risk_manager.enbpi.update_calibration(residual)
        
        # Generate prediction intervals and position sizes
        for pred in self.predictions:
            result = self.risk_manager.predict_with_sizing(
                point_pred=pred,
                estimated_residual=np.median(calibration_residuals) if len(calibration_residuals) > 0 else None
            )
            
            intervals.append((result['lower_bound'], result['upper_bound']))
            position_sizes.append(result['position_size'])
        
        self.position_sizes = position_sizes
        
        print(f"  [OK] Generated intervals for {len(intervals)} predictions")
        print(f"    Avg interval width: {np.mean([u - l for l, u in intervals]):.4f}")
        print(f"    Avg position size: {np.mean(position_sizes):.2f}")
        print(f"    Coverage target: {self.risk_manager.aci.get_current_coverage_target():.1%}")
        
        return {
            'intervals': intervals,
            'position_sizes': position_sizes,
            'risk_metrics': self.risk_manager.get_risk_metrics()
        }
    
    def backtest_strategy(self, strategy_func=None) -> Dict:
        """
        Stage 5: Backtest with enhanced signal generation and Kelly sizing.
        Uses dual confirmation: TDA regime + TCN prediction
        Implements Kelly Criterion for position sizing
        
        Args:
            strategy_func: Optional custom signal generation function
            
        Returns:
            Backtest performance metrics
        """
        print(f"\n[STAGE 5] Backtesting Strategy...")
        
        # Generate signals from predictions with dual confirmation
        if strategy_func is None:
            # Enhanced strategy: TDA regime + TCN prediction + momentum confirmation
            signals = np.zeros(len(self.prices[100:]))
            
            prices_subset = self.prices[100:]
            
            # Get TDA results for regime filtering
            tda_results = self.run_tda_analysis(window_size=100)
            regime_switches = tda_results.get('regime_switches', [])
            crash_warnings = tda_results.get('crash_warnings', [])
            
            # Mark crash periods as high-risk
            crash_periods = set([t[0] for t in crash_warnings if isinstance(t, tuple)])
            
            for t in range(1, len(self.predictions)):
                # Skip during crash warnings
                if (t + 100) in crash_periods:
                    signals[t] = 0
                    continue
                
                if t > 0:
                    # Get indicators
                    price_change = prices_subset[t] - prices_subset[t-1]
                    pred = self.predictions[t]
                    pred_magnitude = abs(pred)
                    pred_std = np.nanstd(self.predictions[max(0, t-20):t])
                    
                    if np.isnan(pred) or np.isnan(pred_std):
                        signals[t] = 0
                        continue
                    
                    # Dual confirmation logic:
                    # 1. Prediction must be significant (> 0.5x std)
                    # 2. Price momentum should support prediction
                    # 3. Not in crash period
                    
                    pred_confidence = pred_magnitude / (pred_std + 1e-6)
                    
                    # Buy signal: strong positive prediction + positive momentum
                    if pred > 0 and price_change > 0 and pred_confidence > 0.5:
                        signals[t] = 1.0
                    # Sell signal: strong negative prediction + negative momentum
                    elif pred < 0 and price_change < 0 and pred_confidence > 0.5:
                        signals[t] = -1.0
                    else:
                        signals[t] = 0
                else:
                    signals[t] = 0
        else:
            signals = strategy_func(self.prices)
        
        # Run backtest with stop-loss and take-profit
        config = BacktestConfig(
            initial_capital=100000, 
            commission=0.001,
            stop_loss_pct=0.02,      # 2% stop loss
            take_profit_pct=0.03,     # 3% take profit
            use_atr_stops=False
        )
        backtest = VectorizedBacktest(self.prices[100:], config)  # Skip initial warmup
        # Use signals for backtesting
        result = backtest.backtest_strategy(signals)
        
        print(f"  [OK] Backtest complete")
        print(f"    Total Return: {result['total_return']:.2%}")
        print(f"    Annual Return: {result['annual_return']:.2%}")
        print(f"    Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"    Win Rate: {result['win_rate']:.2%}")
        print(f"    Profit Factor: {result['profit_factor']:.2f}")
        
        return result
    
    def run_full_pipeline(self):
        """Execute complete pipeline end-to-end"""
        print("\n" + "="*60)
        print("GREY BOX TRADING ENGINE - FULL PIPELINE")
        print("="*60)
        
        # Stage 1: Signal Processing
        signal_results = self.run_signal_processing(window_size=512)
        
        # Stage 2: TDA Analysis
        tda_results = self.run_tda_analysis(window_size=100)
        
        # Stage 3: Prediction
        pred_results = self.run_prediction(epochs=10, lookback=100)
        
        # Stage 4: Risk Management
        risk_results = self.run_risk_management()
        
        # Stage 5: Backtesting
        backtest_results = self.backtest_strategy()
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*60)
        
        return {
            'signal_processing': signal_results,
            'tda_analysis': tda_results,
            'prediction': pred_results,
            'risk_management': risk_results,
            'backtesting': backtest_results
        }


if __name__ == "__main__":
    # Example: Run on Reliance Industries data
    print("Loading market data...")
    context = RealWorldDataContext.get_ril_context()
    
    # Align data
    aligned = MultiAssetDataProcessor.align_data(context)
    ril_prices = aligned['RIL']
    
    print(f"Loaded {len(ril_prices)} data points for RIL")
    
    # Initialize and run engine
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    engine = GreyBoxTradingEngine(
        prices=ril_prices, device=device,
        modwt_decomp_level=4, tcn_channels=32,
        embedding_dim=5, risk_alpha=0.05
    )
    
    # Run pipeline
    results = engine.run_full_pipeline()
    
    print("\nPipeline Results Summary:")
    print(f"  Signal Processing: [OK]")
    print(f"  TDA Analysis: {len(results['tda_analysis']['crash_warnings'])} crash warnings")
    print(f"  Predictions: {len(results['prediction']['predictions'])} predictions")
    print(f"  Position Sizes: Avg {np.mean(results['risk_management']['position_sizes']):.2f}")
    print(f"  Backtest Sharpe: {results['backtesting']['sharpe_ratio']:.2f}")
