# Algorithmic Trading Pipeline: Integrated Grey Box System

## Overview

This is a comprehensive implementation of an **Architectural Blueprint for Next-Generation Algorithmic Trading** that synergizes:

1. **Multi-Resolution Signal Processing** (MODWT)
2. **Topological Geometry** (Persistent Homology / TDA)
3. **Causal Deep Learning** (Temporal Convolutional Networks)
4. **Probabilistic Risk Management** (Conformal Prediction)
5. **Rigorous Backtesting** (Walk-Forward Optimization)

The system is designed to overcome limitations of traditional econometric models (ARIMA, GARCH) that assume stationarity and linearity, while avoiding the black-box nature and robustness issues of standard deep learning approaches.

---

## Architecture

### Stage 1: Multi-Resolution Signal Processing (MODWT)

**File**: `signal_processing/modwt.py`

**Key Components**:
- **CausalMODWT**: Implements Maximal Overlap Discrete Wavelet Transform with causal rolling window approach
- **WaveletVarianceAnalyzer**: Decomposes volatility across timescales

**Why MODWT over DWT?**
- Shift invariance: Properties robust to starting position
- No dyadic length restriction: Works with any sample size
- Accommodates arbitrary frequencies: Critical for tick/bar data re-evaluation

**Key Algorithm: Rolling Window for Causality** (Section 2.3.1)
```
At each time step t:
1. Extract window: X_t = [p_{t-L+1}, ..., p_t] (e.g., L=512 bars)
2. Apply MODWT to entire window
3. Extract ONLY coefficient at t (rightmost point)
4. Discard historical coefficients (calculated in previous iterations)
5. Move to t+1

This ensures p_t depends only on {p_t, p_t-1, ...}, NEVER on p_{t+1}
```

**Usage**:
```python
from signal_processing.modwt import CausalMODWT

modwt = CausalMODWT(wavelet='sym8', decomposition_level=4)
denoised_prices = modwt.wavelet_denoise(prices, window_size=512)
trend, details = modwt.causal_rolling_modwt(prices)
```

---

### Stage 2: Topological Data Analysis (TDA)

**File**: `tda/persistent_homology.py`

**Key Components**:
- **TakensEmbedding**: Phase space reconstruction via time-delay embedding
- **PersistentHomologyAnalyzer**: Crash signal detection via H₁ persistence spikes
- **WassersteinRegimeDetector**: Regime change detection via persistence diagram distances
- **MarketPhaseDetector**: Integrated detector for all TDA analyses

**Takens Embedding** (Section 3.1):
```
V_t = [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]

Where:
- τ: Time delay (lag between components)
- d: Embedding dimension (typically 3-6 for financial data)

This converts univariate price series into geometric point cloud.
```

**Crash Signal Detection** (Section 3.2):
- Monitors H₁ (1-dimensional homology) norm of persistence landscape
- Pre-crash markets exhibit metastability (oscillation between attractors)
- Creates large persistent loops detectable 4-12 weeks before major crashes

**Regime Switching** (Section 3.3):
- Wasserstein distance between consecutive persistence diagrams
- Spikes indicate topological deformation (rules of game changed)
- Used to trigger parameter adaptation or strategy switching

**Usage**:
```python
from tda.persistent_homology import MarketPhaseDetector

detector = MarketPhaseDetector(embedding_dim=5, time_delay=1)
analysis = detector.analyze_window(price_series)

crash_warning = analysis['crash_warning']  # Boolean
regime_switch = analysis['regime_switch']  # Boolean
wasserstein_dist = analysis['wasserstein_distance']  # Float
```

---

### Stage 3: Temporal Convolutional Networks (TCNs)

**File**: `prediction/tcn.py`

**Key Components**:
- **CausalConv1d**: 1D convolution with causal padding (no future info leakage)
- **TemporalBlock**: Residual block with dilated convolutions
- **AttentionLayer**: Multi-head attention for dynamic time-step weighting
- **TemporalConvolutionalNetwork**: Full TCN architecture
- **TCNPredictor**: Training and inference wrapper

**Dilated Causal Convolutions** (Section 4.2):
```
Receptive Field R = 1 + Σ(k-1)·d_i for i in 1..L

Where:
- k: kernel size
- L: number of layers
- d_i: dilation at layer i (exponentially increasing: 1, 2, 4, 8, ...)

Example: 4-layer TCN with kernel=5 achieves receptive field of ~85 steps
```

**Causal Padding** (Section 4.3):
- Left-pad input by (kernel_size - 1) × dilation
- Ensures output at t depends only on inputs ≤ t
- No zeros on right side (prevents future data leakage)

**Attention Mechanism**:
- Multi-head self-attention between TCN blocks
- Weights importance of different time steps
- Higher attention to regime-switch periods detected by TDA

**Usage**:
```python
from prediction.tcn import TCNPredictor

tcn = TCNPredictor(
    input_size=3, output_size=1,
    num_channels=32, kernel_size=5,
    num_levels=4, use_attention=True,
    device='cuda'
)

# Training
loss = tcn.train_step(X_train, y_train)

# Prediction
predictions = tcn.predict(X_test)  # Shape: (batch, seq_len, 1)
```

---

### Stage 4: Risk Management - Conformal Prediction

**File**: `risk_management/conformal_prediction.py`

**Key Components**:
- **ConformalPredictionCalibrator**: Base conformal predictor
- **EnsembleBatchPredictionIntervals (EnbPI)**: Time series conformal prediction
- **AdaptiveConformalInference (ACI)**: PID-like feedback control
- **PositionSizer**: Dynamic position sizing based on intervals
- **RiskManager**: Integrated risk system

**The Exchangeability Problem**:
- Standard Conformal Prediction assumes exchangeability (order doesn't matter)
- Financial time series are non-exchangeable (today depends on yesterday)
- Naive CP results in coverage gaps: 95% CI might only cover 80% during volatility

**EnbPI Solution** (Section 5.2):
```
1. Bootstrap Ensembling: Train B models on bootstrap samples
2. Leave-One-Out (LOO) Residuals: Compute errors from models that didn't see the point
3. Online Aggregation: Update residual distribution as new data arrives
4. Dynamic Width: High-volatility regimes → wider intervals automatically
```

**Adaptive Conformal Inference (ACI)** (Section 5.3):
```
PID-like feedback loop:
- If realized_coverage < target_coverage:
    → Decrease α (alpha) → widen intervals
- If realized_coverage > target_coverage:
    → Increase α → tighten intervals

Ensures 95% CI actually achieves 95% coverage in all market conditions
```

**Position Sizing** (Section 5.3):
```
Size_t ∝ 1 / W_t

Where W_t is interval half-width:
- Narrow interval (high confidence) → larger position
- Wide interval (high uncertainty) → smaller position
- Automatically adapts to market volatility
```

**Usage**:
```python
from risk_management.conformal_prediction import RiskManager

rm = RiskManager(alpha=0.05, num_ensembles=10, base_position_size=1.0)

# Calibrate with historical errors
for y_true, lower, upper in calibration_data:
    residual = abs(y_true - (lower + upper) / 2)
    rm.enbpi.update_calibration(residual)

# Make prediction with intervals and sizing
result = rm.predict_with_sizing(point_pred=100.5, estimated_residual=1.2)
print(result['lower_bound'])      # ₹98.2
print(result['upper_bound'])      # ₹102.8
print(result['position_size'])    # 0.85 units
print(result['target_coverage'])  # 95%
```

---

### Stage 5: Backtesting & Optimization

**File**: `backtesting/vectorized_backtest.py`

**Key Components**:
- **VectorizedBacktest**: Efficient vectorized backtesting
- **WalkForwardOptimizer**: Robust parameter validation
- **MetricsCalculator**: Comprehensive performance metrics

**Vectorized Backtesting**:
- Processes entire strategy across multiple parameter combinations
- NumPy/Numba acceleration: millions of permutations in seconds
- Essential for testing parameter stability across markets

**Walk-Forward Optimization** (Section 6.1):
```
For each time window:
1. In-Sample (Training): Optimize parameters on 1000 bars
2. Out-of-Sample (Testing): Evaluate on subsequent 250 bars
3. Measure: Does best in-sample parameters perform well OOS?

Prevents overfitting and validates parameter robustness
```

**Performance Metrics**:
- Total Return, Annual Return
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, Win Rate, Profit Factor

**Usage**:
```python
from backtesting.vectorized_backtest import VectorizedBacktest, WalkForwardOptimizer

# Simple backtest
config = BacktestConfig(initial_capital=100000, commission=0.001)
backtest = VectorizedBacktest(prices, config)
signals = backtest.generate_signals(my_signal_func)
result = backtest.backtest_strategy(signals)

print(f"Sharpe: {result['sharpe_ratio']:.2f}")
print(f"Return: {result['total_return']:.2%}")

# Walk-forward optimization
wfo = WalkForwardOptimizer(prices, config)
wfo_results = wfo.walk_forward_analysis(
    my_signal_func,
    {'window': [15, 20, 25], 'threshold': [1.0, 1.5, 2.0]},
    in_sample_size=1000, out_of_sample_size=250
)
```

---

### Main Pipeline Integration

**File**: `main_pipeline.py`

**GreyBoxTradingEngine**: Orchestrates all 5 stages

```python
from main_pipeline import GreyBoxTradingEngine

engine = GreyBoxTradingEngine(
    prices=ril_prices,
    device='cuda',
    modwt_decomp_level=4,
    tcn_channels=32,
    embedding_dim=5,
    risk_alpha=0.05
)

# Run complete pipeline
results = engine.run_full_pipeline()

# Access individual stage results
signal_results = results['signal_processing']
tda_results = results['tda_analysis']
predictions = results['prediction']['predictions']
risk_metrics = results['risk_management']['risk_metrics']
backtest_sharpe = results['backtesting']['sharpe_ratio']
```

**Pipeline Stages**:
1. `run_signal_processing()`: MODWT denoise + volatility analysis
2. `run_tda_analysis()`: Crash detection + regime switching
3. `run_prediction()`: TCN training + price forecasting
4. `run_risk_management()`: Conformal intervals + position sizing
5. `backtest_strategy()`: Validation + performance metrics

---

## Streamlit Dashboard

**File**: `dashboard/app.py`

Interactive monitoring system with 5 tabs:

1. **Real-Time Overview**: Price, confidence, position size, regime status
2. **Signal Processing**: MODWT decomposition visualization, boundary handling
3. **Topological Analysis**: TDA regime indicator (green/yellow/red), persistence metrics
4. **Predictions & Sizing**: Confidence-to-position mapping, prediction intervals table
5. **Backtesting Results**: Walk-forward performance, parameter stability analysis

**Run Dashboard**:
```bash
streamlit run dashboard/app.py
```

**Features**:
- 🛑 Manual kill switch for emergency trading halt
- 📊 Real-time metric updates
- 🧬 Topological regime visualization as traffic light
- 🎯 Position sizing logic with confidence scores
- 📈 Walk-forward optimization results with parameter stability

---

## Case Study: Reliance Industries (2026 Outlook)

**File**: `utils/data_utils.py` → `RealWorldDataContext.get_ril_context()`

**Market Context** (Section 7):
- **Asset**: Reliance Industries Ltd (RIL) - Indian conglomerate
- **Linked Assets**: Brent Crude Futures, USD/INR exchange rate
- **Correlation**: Positive to crude prices (O2C segment benefits), but non-linear

**Algorithm Execution Scenario**:
1. **Regime Detection (TDA)**: Oil plummets → compute Wasserstein distance between RIL-Oil correlation structure and historical baseline. If spike, signals "decoupling" - RIL now valued as Tech/Consumer (Jio) rather than Energy.

2. **Trend Filtering (MODWT)**: Wavelet separates daily oil crash noise from underlying trend. Approximation coefficients stable → drop identified as noise, not trend reversal.

3. **Prediction (TCN + CP)**: TCN attends to "decoupling" signal from TDA. Predicts stable price. Conformal Prediction widens interval due to volatility in exogenous variable (Oil).

4. **Trade Decision**: Hold long position but reduce size (wider uncertainty band). Protects capital against tail-risk while riding volatility.

**Usage**:
```python
from utils.data_utils import RealWorldDataContext, MultiAssetDataProcessor

# Fetch RIL + correlated assets
context = RealWorldDataContext.get_ril_context(end_date='2026-01-20')

# Align on common dates
aligned = MultiAssetDataProcessor.align_data(context)

# Create multivariate features
X, feature_names = MultiAssetDataProcessor.create_multivariate_features(
    aligned, n_lags=10
)
```

---

## Installation & Setup

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, Pandas
- PyWavelets, giotto-tda
- Streamlit, Plotly

### Installation
```bash
cd algorithmic_trading_pipeline
pip install -r requirements.txt
```

### Quick Start
```python
# Example: Run complete pipeline on synthetic data
import numpy as np
from main_pipeline import GreyBoxTradingEngine

# Generate synthetic prices
np.random.seed(42)
prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 1000)))

# Initialize engine
engine = GreyBoxTradingEngine(prices=prices, device='cpu')

# Run full pipeline
results = engine.run_full_pipeline()

print(f"Backtest Sharpe Ratio: {results['backtesting']['sharpe_ratio']:.2f}")
print(f"Crash warnings: {len(results['tda_analysis']['crash_warnings'])}")
print(f"Avg position size: {np.mean(results['risk_management']['position_sizes']):.2f}")
```

---

## Critical Implementation Details

### 1. Causality Enforcement

All components strictly enforce causality to prevent look-ahead bias:

**MODWT**: Rolling window with coefficient extraction at t only
**TCN**: Causal padding ensures output depends only on ≤t inputs
**Conformal**: LOO residuals computed on unseen data
**Backtesting**: No data alignment beyond contemporaneous

### 2. Time-Frequency Localization

MODWT provides both:
- **Time localization**: Wavelet identify WHEN events occur
- **Frequency localization**: Wavelet identify WHAT type of movement

Unlike Fourier (which globally averages frequency), MODWT is ideal for:
- Flash crash detection (sudden high-frequency volatility)
- Trend identification (low-frequency components)
- Microstructure noise filtering

### 3. Topological Robustness

TDA captures market structure independent of:
- Absolute price levels
- Linear scalings
- Distributional assumptions

Example: Two markets with different correlation matrices but same topological structure get treated equivalently by TDA.

### 4. Deep Learning Interpretability

TCN architecture provides transparency:
- **Receptive field**: Explicitly calculated
- **Causal structure**: Guaranteed by design
- **Attention weights**: Human-readable importance scores
- **No black box**: All computations deterministic and auditable

### 5. Risk Management Rigor

Conformal Prediction guarantees:
- **Coverage guarantee**: 95% CI covers true value ≥95% of time (proven)
- **Distribution-free**: No normality assumptions
- **Online adaptation**: Handles regime changes automatically
- **Finite sample**: No asymptotic approximations needed

---

## Limitations & Future Work

### Current Limitations
1. Computational cost: Full MODWT per tick requires optimization
2. Parameter sensitivity: Wavelet type, embedding dim, TCN architecture need careful tuning
3. Limited to univariate + exogenous variables (no microstructure data)
4. Historical backtesting cannot capture tail events outside training range

### Future Enhancements
1. **GPU-accelerated wavelets**: CuPy implementation for real-time processing
2. **Multimodal learning**: Incorporate order book dynamics, sentiment signals
3. **Adaptive architecture**: Meta-learning to optimize component parameters
4. **Quantum computing**: Potential for TDA persistent homology acceleration
5. **Explainability layer**: SHAP/LIME integration for trade decision attribution

---

## References

All citations reference research papers and industry resources dated January 20, 2026:

**Wavelet Analysis**
- MODWT-based decomposition (m-hikari.com, 2011)
- Boundary effect elimination techniques (GitHub, Quant Stack Exchange)

**Topological Data Analysis**
- Persistent homology for crash detection (arXiv, MDPI, Medium)
- Wasserstein distance for regime detection (arXiv, AIMS Press)

**Deep Learning**
- Temporal Convolutional Networks (IEEE, Medium)
- Attention mechanisms in time series (MDPI)

**Uncertainty Quantification**
- Conformal Prediction for time series (arXiv, NeurIPS, MLPR)
- Adaptive Conformal Inference (NeurIPS 2021)

---

## License & Disclaimer

**Educational Purpose Only**: This codebase is for research, educational, and paper purposes only.

**No Guarantee**: Past performance does not guarantee future results. Algorithmic trading involves substantial risk of loss.

**Professional Use**: Before live trading, conduct extensive backtesting, stress-testing, and risk assessment with qualified professionals.

**Regulatory Compliance**: Ensure compliance with all applicable financial regulations in your jurisdiction.

---

## Contact & Contributions

For questions, issues, or contributions, refer to the main documentation or contact the development team.

Last Updated: January 20, 2026
System Version: 1.0 (Beta)
Status: Production-Ready for Backtesting | Research Phase for Live Trading
