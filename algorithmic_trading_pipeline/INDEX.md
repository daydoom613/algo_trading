# Algorithmic Trading Pipeline - File Index

## 📖 Documentation Files
- **README.md** (17.5 KB) - Comprehensive documentation with theory, architecture, and usage guide
- **IMPLEMENTATION_SUMMARY.md** (14.2 KB) - Executive summary of implementation, components, and quick start

## 🎯 Main Entry Points
- **main_pipeline.py** (15.1 KB) - GreyBoxTradingEngine orchestrator (5-stage pipeline)
- **examples.py** (8.6 KB) - Quick start with 6 self-contained examples

## 🔊 Signal Processing Module
`signal_processing/`
- **modwt.py** (10.1 KB) - CausalMODWT with rolling window approach
  - CausalMODWT class: causal_rolling_modwt(), wavelet_denoise()
  - WaveletVarianceAnalyzer: volatility decomposition

## 🧬 Topological Data Analysis Module
`tda/`
- **persistent_homology.py** (14.0 KB) - Phase space reconstruction & crash detection
  - TakensEmbedding: Phase space reconstruction with optimal embedding
  - PersistentHomologyAnalyzer: H_1 spikes for crash warnings
  - WassersteinRegimeDetector: Regime switching via persistence diagrams
  - MarketPhaseDetector: Integrated detector

## 🎯 Prediction Module
`prediction/`
- **tcn.py** (13.6 KB) - Temporal Convolutional Networks
  - CausalConv1d: Causal padding implementation
  - TemporalBlock: Residual blocks with dilation
  - AttentionLayer: Multi-head attention
  - TemporalConvolutionalNetwork: Full architecture
  - TCNPredictor: Training/inference wrapper

## 📊 Risk Management Module
`risk_management/`
- **conformal_prediction.py** (16.0 KB) - Uncertainty quantification
  - ConformalPredictionCalibrator: Base calibration
  - EnsembleBatchPredictionIntervals (EnbPI): Time series conformal
  - AdaptiveConformalInference (ACI): PID feedback control
  - PositionSizer: Dynamic position sizing (Size ∝ 1/interval_width)
  - RiskManager: Integrated system

## 📈 Backtesting Module
`backtesting/`
- **vectorized_backtest.py** (16.8 KB) - Efficient backtesting
  - VectorizedBacktest: Vectorized strategy testing
  - WalkForwardOptimizer: Robust parameter validation
  - MetricsCalculator: Comprehensive performance metrics

## 📚 Utilities Module
`utils/`
- **data_utils.py** (10.1 KB) - Data handling
  - DataFetcher: yfinance integration
  - DataPreprocessor: Feature engineering
  - MultiAssetDataProcessor: Multi-asset alignment
  - RealWorldDataContext: RIL + Brent + USD/INR example

## 🎨 Dashboard Module
`dashboard/`
- **app.py** (14.8 KB) - Streamlit interactive system
  - 5 visualization tabs
  - Real-time metrics
  - MODWT plots
  - TDA regime indicator
  - Position sizing logic
  - Walk-forward results

## 📦 Package Files
- **__init__.py** files in each module for Python package structure:
  - signal_processing/__init__.py
  - tda/__init__.py
  - prediction/__init__.py
  - risk_management/__init__.py
  - backtesting/__init__.py
  - utils/__init__.py
  - dashboard/__init__.py

## 🔧 Configuration Files
- **requirements.txt** (261 B) - All dependencies

---

## 📊 Code Statistics

| Module | File | Size | Purpose |
|--------|------|------|---------|
| Signal Processing | modwt.py | 10.1 KB | MODWT with causality |
| TDA | persistent_homology.py | 14.0 KB | Phase space + crash detection |
| Prediction | tcn.py | 13.6 KB | Dilated causal CNN |
| Risk Mgmt | conformal_prediction.py | 16.0 KB | EnbPI + ACI + sizing |
| Backtesting | vectorized_backtest.py | 16.8 KB | Vectorized testing + WFO |
| Utils | data_utils.py | 10.1 KB | Data handling |
| Dashboard | app.py | 14.8 KB | Streamlit UI |
| Pipeline | main_pipeline.py | 15.1 KB | Orchestrator |
| Examples | examples.py | 8.6 KB | 6 quick examples |
| **Total** | **9 files** | **109 KB** | **Production ready** |

---

## 🚀 Getting Started

### Step 1: Installation
```bash
cd d:\trading\algorithmic_trading_pipeline
pip install -r requirements.txt
```

### Step 2: Run Examples (5-10 minutes)
```bash
python examples.py
```

### Step 3: Run Full Pipeline (2-3 minutes)
```bash
python main_pipeline.py
```

### Step 4: Launch Dashboard
```bash
streamlit run dashboard/app.py
```

Open browser: http://localhost:8501

---

## 🎯 Component Architecture

```
Input Prices
     ↓
[STAGE 1] MODWT Signal Processing
     ↓ (denoised prices)
[STAGE 2] TDA Phase Detection
     ↓ (crash warnings, regime switches)
[STAGE 3] TCN Prediction (with attention to TDA signals)
     ↓ (price predictions)
[STAGE 4] Conformal Prediction & Position Sizing
     ↓ (prediction intervals + sizes)
[STAGE 5] Backtesting & Walk-Forward Optimization
     ↓
Output: Sharpe ratio, drawdown, win rate, etc.
```

---

## 📝 Key Files to Understand

1. **For Signal Processing**: `signal_processing/modwt.py`
   - Understand: Causal rolling window, Symlet-8 wavelet
   - Function: `causal_rolling_modwt()`

2. **For TDA**: `tda/persistent_homology.py`
   - Understand: Takens embedding, persistent homology
   - Function: `detect_crash_signal()`, `detect_regime_switch()`

3. **For TCN**: `prediction/tcn.py`
   - Understand: Dilated causal convolutions, attention
   - Class: `TemporalConvolutionalNetwork`

4. **For Risk**: `risk_management/conformal_prediction.py`
   - Understand: EnbPI, ACI feedback, position sizing
   - Function: `predict_with_sizing()`

5. **For Backtesting**: `backtesting/vectorized_backtest.py`
   - Understand: Walk-forward optimization
   - Function: `walk_forward_analysis()`

6. **For Integration**: `main_pipeline.py`
   - Understand: All 5 stages working together
   - Class: `GreyBoxTradingEngine`

---

## 💡 Usage Examples

### Quick Test (1 minute)
```python
from main_pipeline import GreyBoxTradingEngine
import numpy as np

prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 500)))
engine = GreyBoxTradingEngine(prices)
results = engine.run_full_pipeline()
print(f"Sharpe: {results['backtesting']['sharpe_ratio']:.2f}")
```

### Full Analysis (5 minutes)
```bash
python main_pipeline.py
```

### Explore Components
```bash
python examples.py
```

### Monitor Real-Time
```bash
streamlit run dashboard/app.py
```

---

## ✨ What's Implemented

✅ Causal MODWT signal processing
✅ Takens phase space embedding
✅ Persistent homology crash detection
✅ Wasserstein regime switching
✅ Dilated causal TCN prediction
✅ Multi-head attention mechanism
✅ Ensemble batch prediction intervals
✅ Adaptive conformal inference
✅ Dynamic position sizing
✅ Vectorized backtesting
✅ Walk-forward optimization
✅ Interactive Streamlit dashboard
✅ Real-world data integration (RIL example)
✅ Comprehensive documentation

---

## 📚 Additional Resources

- **README.md**: Detailed theory and architecture
- **IMPLEMENTATION_SUMMARY.md**: Executive overview
- **examples.py**: 6 runnable examples
- **Each module docstring**: Detailed API documentation
- **Code comments**: Implementation details

---

## 🎓 Learning Path

1. **Theory** → Read README.md sections 2-5
2. **Examples** → Run examples.py to see each component
3. **Integration** → Review main_pipeline.py for orchestration
4. **Practice** → Modify examples.py with your own data
5. **Monitor** → Use dashboard/app.py for real-time visualization
6. **Production** → Customize and integrate with your infrastructure

---

**Status**: ✅ Implementation Complete | Ready for Use
**Date**: January 20, 2026
**Version**: 1.0 (Production)
