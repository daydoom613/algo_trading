"""
Risk Management Layer: Conformal Prediction for Uncertainty Quantification
Implements Ensemble Batch Prediction Intervals (EnbPI) and Adaptive Conformal Inference (ACI)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import deque
from scipy import stats


class ConformalPredictionCalibrator:
    """
    Base calibration for conformal prediction on time series.
    Handles the fundamental exchangeability problem in financial data.
    """
    
    def __init__(self, alpha: float = 0.05, window_size: int = 100):
        """
        Initialize calibrator.
        
        Args:
            alpha: Miscoverage level (1 - confidence). E.g., 0.05 for 95% CI.
            window_size: Size of rolling window for calibration statistics
        """
        self.alpha = alpha
        self.window_size = window_size
        self.residuals = deque(maxlen=window_size)
        self.is_calibrated = False
    
    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calibrate on historical prediction errors.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        residuals = np.abs(y_true - y_pred)
        self.residuals.extend(residuals)
        
        if len(self.residuals) >= self.window_size // 2:
            self.is_calibrated = True
    
    def get_prediction_interval(self, point_pred: float) -> Tuple[float, float]:
        """
        Get prediction interval from calibrated residuals.
        
        Returns:
            (lower_bound, upper_bound)
        """
        if not self.is_calibrated:
            # Use default interval until calibrated
            return point_pred - 2.0, point_pred + 2.0
        
        # Quantile-based interval
        residuals_array = np.array(list(self.residuals))
        
        # Coverage adjustment: slightly increase to ensure coverage
        adjusted_alpha = min(self.alpha + 0.01, 0.5)
        
        quantile_val = np.quantile(residuals_array, 1 - adjusted_alpha)
        
        lower = point_pred - quantile_val
        upper = point_pred + quantile_val
        
        return lower, upper


class EnsembleBatchPredictionIntervals:
    """
    Ensemble Batch Prediction Intervals (EnbPI) for time series.
    
    Specifically designed for non-exchangeable sequential data.
    Uses bootstrap ensembles and Leave-One-Out (LOO) residuals.
    """
    
    def __init__(self, alpha: float = 0.05, num_ensembles: int = 10,
                 calibration_size: int = 200):
        """
        Initialize EnbPI.
        
        Args:
            alpha: Miscoverage level
            num_ensembles: Number of bootstrap ensemble models
            calibration_size: Size of calibration set
        """
        self.alpha = alpha
        self.num_ensembles = num_ensembles
        self.calibration_size = calibration_size
        
        # Store calibration residuals
        self.calibration_residuals = deque(maxlen=calibration_size)
        
        # LOO prediction errors
        self.loo_residuals = deque(maxlen=calibration_size)
    
    def update_calibration(self, residual: float):
        """
        Update calibration set with new residual.
        This happens online as new data arrives.
        
        Args:
            residual: Prediction error (|y_true - y_pred|)
        """
        self.calibration_residuals.append(residual)
        self.loo_residuals.append(residual)
    
    def get_interval_width(self, regressed_residual: Optional[float] = None) -> float:
        """
        Calculate interval width based on calibrated residuals.
        
        If prediction error increases (volatile regime), intervals automatically widen.
        This is the key adaptive feature of EnbPI for non-stationary data.
        
        Args:
            regressed_residual: Optional residual from regression/model
            
        Returns:
            Interval half-width (distance from point estimate)
        """
        if len(self.calibration_residuals) < 10:
            # Not yet calibrated
            return 2.0
        
        residuals_array = np.array(list(self.calibration_residuals))
        
        # Quantile for coverage
        # n+1 quantile for conformal prediction (slightly conservative)
        n = len(residuals_array)
        quantile_index = int(np.ceil((n + 1) * (1 - self.alpha)) / n)
        quantile_index = min(quantile_index, n - 1)
        
        sorted_residuals = np.sort(residuals_array)
        base_width = sorted_residuals[quantile_index]
        
        # If regressed residual provided, blend it
        if regressed_residual is not None:
            base_width = 0.6 * base_width + 0.4 * regressed_residual
        
        return base_width
    
    def predict_with_intervals(self, point_predictions: np.ndarray,
                              regressed_residuals: Optional[np.ndarray] = None) -> dict:
        """
        Get prediction intervals for multiple predictions.
        
        Args:
            point_predictions: Array of point estimates
            regressed_residuals: Optional array of residuals from regression
            
        Returns:
            Dictionary with lower, upper bounds and widths
        """
        if len(point_predictions) == 0:
            return {'lower': [], 'upper': [], 'width': []}
        
        widths = []
        lowers = []
        uppers = []
        
        for i, pred in enumerate(point_predictions):
            reg_res = regressed_residuals[i] if regressed_residuals is not None else None
            width = self.get_interval_width(reg_res)
            
            lower = pred - width
            upper = pred + width
            
            widths.append(width)
            lowers.append(lower)
            uppers.append(upper)
        
        return {
            'lower': np.array(lowers),
            'upper': np.array(uppers),
            'width': np.array(widths)
        }


class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (ACI) wrapping EnbPI.
    
    Uses PID-like control to adjust coverage level dynamically.
    If coverage < target, widen intervals. If coverage > target, tighten.
    """
    
    def __init__(self, target_coverage: float = 0.95, learning_rate: float = 0.01,
                 lookback_window: int = 50):
        """
        Initialize ACI controller.
        
        Args:
            target_coverage: Target coverage probability (e.g., 0.95 for 95%)
            learning_rate: Step size for alpha adjustment
            lookback_window: Number of recent predictions to evaluate coverage
        """
        self.target_coverage = target_coverage
        self.target_alpha = 1 - target_coverage
        self.learning_rate = learning_rate
        self.lookback_window = lookback_window
        
        # Current alpha (miscoverage level)
        self.alpha = self.target_alpha
        
        # Track coverage
        self.coverage_history = deque(maxlen=lookback_window)
    
    def update_coverage(self, y_true: float, lower: float, upper: float):
        """
        Update coverage tracking.
        
        Args:
            y_true: True value
            lower: Prediction interval lower bound
            upper: Prediction interval upper bound
        """
        is_covered = (lower <= y_true <= upper)
        self.coverage_history.append(float(is_covered))
    
    def adjust_alpha(self) -> float:
        """
        Adjust alpha based on recent coverage (PID-like feedback).
        
        Returns:
            Updated alpha value
        """
        if len(self.coverage_history) < self.lookback_window // 2:
            return self.alpha
        
        recent_coverage = np.mean(list(self.coverage_history))
        coverage_error = recent_coverage - self.target_coverage
        
        # Simple proportional control: if coverage too low, decrease alpha (widen intervals)
        alpha_adjustment = -coverage_error * self.learning_rate
        
        self.alpha = np.clip(self.alpha + alpha_adjustment, 0.01, 0.5)
        
        return self.alpha
    
    def get_current_alpha(self) -> float:
        """Get current miscoverage level"""
        return self.alpha
    
    def get_current_coverage_target(self) -> float:
        """Get current coverage probability (1 - alpha)"""
        return 1 - self.alpha


class PositionSizer:
    """
    Position sizing based on prediction interval width.
    
    Implements: Size_t ∝ 1 / W_t
    
    When model is uncertain (wide intervals), reduce position size.
    When model is confident (narrow intervals), increase size.
    """
    
    def __init__(self, base_size: float = 1.0, min_size: float = 0.1,
                 max_size: float = 2.0):
        """
        Initialize position sizer.
        
        Args:
            base_size: Reference position size (e.g., 1.0 = 1 standard unit)
            min_size: Minimum position size
            max_size: Maximum position size
        """
        self.base_size = base_size
        self.min_size = min_size
        self.max_size = max_size
        
        # Reference width for base size
        self.reference_width = 1.0
    

    def set_reference_width(self, widths: np.ndarray):
        """
        Calibrate reference width from historical intervals.
        
        Args:
            widths: Array of historical interval widths
        """
        self.reference_width = np.median(widths)
    
    def calculate_size(self, interval_width: float) -> float:
        """
        Calculate position size for given prediction interval width.
        
        Size = base_size * (reference_width / current_width)
        
        Args:
            interval_width: Half-width of prediction interval
            
        Returns:
            Position size (clipped to [min_size, max_size])
        """
        if interval_width <= 0:
            return self.min_size
        
        size = self.base_size * (self.reference_width / interval_width)
        
        # Clip to limits
        size = np.clip(size, self.min_size, self.max_size)
        
        return size
    
    def calculate_sizes_batch(self, interval_widths: np.ndarray) -> np.ndarray:
        """
        Calculate position sizes for multiple intervals.
        
        Args:
            interval_widths: Array of prediction interval widths
            
        Returns:
            Array of position sizes
        """
        sizes = np.array([self.calculate_size(w) for w in interval_widths])
        return sizes


class RiskManager:
    """
    Integrated risk management system combining all components.
    """
    
    def __init__(self, alpha: float = 0.05, num_ensembles: int = 10,
                 base_position_size: float = 1.0):
        """
        Initialize risk manager.
        
        Args:
            alpha: Initial miscoverage level
            num_ensembles: Number of ensemble models
            base_position_size: Base position size for sizing calculations
        """
        self.alpha = alpha
        
        # Conformal prediction components
        self.enbpi = EnsembleBatchPredictionIntervals(
            alpha=alpha, num_ensembles=num_ensembles
        )
        
        self.aci = AdaptiveConformalInference(
            target_coverage=1 - alpha, learning_rate=0.01
        )
        
        # Position sizing
        self.sizer = PositionSizer(base_size=base_position_size)
        
        # Risk metrics tracking
        self.prediction_history = []
        self.interval_history = []
        self.coverage_history = []
    
    def update_with_realization(self, y_true: float, lower: float, upper: float,
                                interval_width: float):
        """
        Update risk manager with realized prediction outcome.
        
        Args:
            y_true: True observed value
            lower: Prediction interval lower bound
            upper: Prediction interval upper bound
            interval_width: Half-width of interval
        """
        # Update calibration
        residual = np.abs(y_true - (lower + upper) / 2)  # Center of interval
        self.enbpi.update_calibration(residual)
        
        # Update ACI
        self.aci.update_coverage(y_true, lower, upper)
        self.aci.adjust_alpha()
        
        # Track history
        is_covered = (lower <= y_true <= upper)
        self.coverage_history.append(is_covered)
    
    def predict_with_sizing(self, point_pred: float,
                           estimated_residual: Optional[float] = None) -> dict:
        """
        Make prediction with interval and position size.
        
        Args:
            point_pred: Point prediction
            estimated_residual: Estimated prediction error (optional)
            
        Returns:
            Dictionary with prediction, interval, position size, and coverage info
        """
        # Get interval
        width = self.enbpi.get_interval_width(estimated_residual)
        lower = point_pred - width
        upper = point_pred + width
        
        # Calculate position size
        position_size = self.sizer.calculate_size(width)
        
        # Get current coverage
        current_coverage = self.aci.get_current_coverage_target()
        
        return {
            'point_prediction': point_pred,
            'lower_bound': lower,
            'upper_bound': upper,
            'interval_width': width,
            'position_size': position_size,
            'alpha': self.aci.get_current_alpha(),
            'target_coverage': current_coverage,
            'confidence_interval': f"[{lower:.4f}, {upper:.4f}]"
        }
    
    def get_risk_metrics(self) -> dict:
        """
        Get summary risk metrics.
        
        Returns:
            Dictionary with realized coverage, alpha, and other metrics
        """
        if len(self.coverage_history) == 0:
            return {
                'realized_coverage': None, 
                'current_alpha': self.aci.alpha,
                'target_coverage': self.aci.target_coverage
            }
        
        realized_coverage = np.mean(self.coverage_history[-50:])  # Last 50 predictions
        
        return {
            'realized_coverage': realized_coverage,
            'target_coverage': self.aci.target_coverage,
            'current_alpha': self.aci.get_current_alpha(),
            'total_predictions': len(self.coverage_history),
            'calibration_residuals_count': len(self.enbpi.calibration_residuals)
        }


if __name__ == "__main__":
    # Example usage
    rm = RiskManager(alpha=0.05, num_ensembles=10, base_position_size=1.0)
    
    # Simulate predictions and realizations
    np.random.seed(42)
    for i in range(100):
        # Prediction
        point_pred = 100 + i * 0.1
        estimated_residual = 1.0 + 0.5 * np.sin(i / 10)
        
        result = rm.predict_with_sizing(point_pred, estimated_residual)
        
        # Realization (with some noise)
        y_true = point_pred + np.random.normal(0, 0.8)
        
        rm.update_with_realization(
            y_true, result['lower_bound'], result['upper_bound'],
            result['interval_width']
        )
        
        if i % 20 == 0:
            print(f"Step {i}: Pred={point_pred:.2f}, Size={result['position_size']:.2f}, "
                  f"α={result['alpha']:.3f}")
    
    # Final metrics
    metrics = rm.get_risk_metrics()
    print("\nFinal Risk Metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
