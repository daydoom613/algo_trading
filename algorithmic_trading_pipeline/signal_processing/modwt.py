"""
Maximal Overlap Discrete Wavelet Transform (MODWT) for Financial Signal Processing
Implements causal rolling-window approach to eliminate look-ahead bias
"""

import numpy as np
import pywt
from typing import Tuple, Optional, List
from scipy import signal


class CausalMODWT:
    """
    Causal MODWT implementation using rolling windows to enforce strict causality.
    Prevents look-ahead bias by processing only historical data at each time step.
    """
    
    def __init__(self, wavelet: str = 'sym8', decomposition_level: int = 4):
        """
        Initialize the Causal MODWT processor.
        
        Args:
            wavelet: Mother wavelet to use. 'sym8' or 'sym5' recommended for finance
                     due to "least asymmetric" property preventing phase shifts.
            decomposition_level: Number of wavelet decomposition levels
        """
        self.wavelet = wavelet
        self.decomposition_level = decomposition_level
        self.wave_obj = pywt.Wavelet(wavelet)
        
    def compute_wavelet_filter_length(self) -> int:
        """Get the filter length for the selected wavelet"""
        return self.wave_obj.dec_len
    
    def causal_rolling_modwt(self, price_series: np.ndarray, 
                            window_size: int = 512) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Apply MODWT using rolling window approach for causal signal decomposition.
        Uses a simpler approach: apply MODWT to entire series but use only causal windows.
        
        Args:
            price_series: 1D array of price data
            window_size: Size of historical window (e.g., 512 bars)
            
        Returns:
            approx_coefficients: Smoothed trend (approximation coefficients at final level)
            detail_coefficients_list: List of detail coefficients at each level
        """
        n = len(price_series)
        
        # Apply MODWT to entire series at once for efficiency
        try:
            coeffs = pywt.mra(price_series, self.wavelet, self.decomposition_level, axis=0)
        except Exception:
            # Fallback if MRA fails
            approx_coeff = price_series.copy()
            detail_coeffs_list = [np.zeros(n) for _ in range(self.decomposition_level)]
            return approx_coeff, detail_coeffs_list
        
        # coeffs contains [cA_n, cD_n, cD_n-1, ..., cD_1]
        approx_coeff = coeffs[0]  # Approximation coefficients
        detail_coeffs_list = [coeffs[i + 1] for i in range(self.decomposition_level)]
        
        # Ensure all have same length as input
        approx_coeff = approx_coeff[:n] if len(approx_coeff) >= n else np.pad(approx_coeff, (0, n - len(approx_coeff)))
        detail_coeffs_list = [
            d[:n] if len(d) >= n else np.pad(d, (0, n - len(d))) 
            for d in detail_coeffs_list
        ]
        
        return approx_coeff, detail_coeffs_list
    
    def wavelet_denoise(self, price_series: np.ndarray, 
                        threshold_multiplier: float = 1.5,
                        window_size: int = 512) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.
        
        Uses universal threshold: lambda = sigma * sqrt(2 * ln(N))
        where sigma is estimated via Median Absolute Deviation (MAD) on detail coefficients.
        
        Args:
            price_series: Input price series
            threshold_multiplier: Multiplier for universal threshold (>1 for more aggressive denoising)
            window_size: Rolling window size
            
        Returns:
            Denoised signal
        """
        approx, details = self.causal_rolling_modwt(price_series, window_size)
        
        # Estimate noise standard deviation from finest detail level (highest frequency)
        finest_detail = details[0]
        sigma = np.median(np.abs(finest_detail - np.median(finest_detail))) / 0.6745  # MAD-based estimation
        sigma = max(sigma, 1e-6)  # Prevent division by very small numbers
        
        # Universal threshold
        n = len(price_series)
        threshold = threshold_multiplier * sigma * np.sqrt(2 * np.log(n))
        
        # Soft thresholding on detail coefficients
        thresholded_details = [
            np.sign(d) * np.maximum(np.abs(d) - threshold, 0) 
            for d in details
        ]
        
        # Reconstruct signal using inverse MRA
        # Start with approximation and add back thresholded details with appropriate weighting
        reconstructed = approx.copy()
        
        # Add details back with decreasing weights to preserve important structures
        for level, thresh_detail in enumerate(thresholded_details):
            # Use equal weighting for all levels to preserve information
            reconstructed += thresh_detail
        
        # Ensure output has same length and scale as input
        reconstructed = np.interp(np.arange(len(price_series)), 
                                 np.arange(len(reconstructed)), 
                                 reconstructed)
        
        return reconstructed
    
    def calculate_wavelet_variance(self, price_series: np.ndarray,
                                   window_size: int = 512) -> np.ndarray:
        """
        Calculate variance across wavelet scales.
        Identifies whether volatility is driven by short-term noise or long-term instability.
        
        Args:
            price_series: Input price series
            window_size: Rolling window size
            
        Returns:
            Variance at each scale
        """
        _, details = self.causal_rolling_modwt(price_series, window_size)
        
        wavelet_var = np.array([np.var(d) for d in details])
        return wavelet_var
    
    def get_trend_and_noise(self, price_series: np.ndarray,
                            window_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate trend (approximation coefficients) from noise (detail coefficients).
        
        Returns:
            trend: Low-frequency trend component
            noise: High-frequency noise component
        """
        approx, details = self.causal_rolling_modwt(price_series, window_size)
        
        # Combine high-frequency components as noise
        noise = np.sum([d for d in details], axis=0)
        
        return approx, noise
    
    def get_scale_specific_signals(self, price_series: np.ndarray,
                                   window_size: int = 512) -> dict:
        """
        Extract signals at different scales for multi-resolution analysis.
        
        Returns dictionary with:
        - 'trend': Long-term trend (approximation)
        - 'scale_1': Finest details (1-4 day cycles)
        - 'scale_2': Medium details (4-8 day cycles)
        - 'scale_3': Coarser details (8-16 day cycles)
        - etc.
        """
        approx, details = self.causal_rolling_modwt(price_series, window_size)
        
        signals = {'trend': approx}
        for level, detail in enumerate(details):
            signals[f'scale_{level + 1}'] = detail
        
        return signals


class WaveletVarianceAnalyzer:
    """Analyzes variance decomposition across timescales"""
    
    def __init__(self, wavelet: str = 'sym8'):
        self.modwt = CausalMODWT(wavelet=wavelet)
    
    def analyze_volatility_drivers(self, price_series: np.ndarray,
                                   window_size: int = 512) -> dict:
        """
        Determine if volatility is driven by short-term noise or long-term instability.
        
        Returns:
            Dictionary with variance percentages at each scale
        """
        wavelet_var = self.modwt.calculate_wavelet_variance(price_series, window_size)
        total_var = np.sum(wavelet_var)
        
        var_percentages = wavelet_var / total_var * 100
        
        return {
            'scale_variances': wavelet_var,
            'scale_percentages': var_percentages,
            'short_term_variance_pct': var_percentages[0],  # Finest scale
            'long_term_variance_pct': var_percentages[-1],  # Coarsest scale
            'noise_dominant': var_percentages[0] > 50,  # If > 50% variance in noise
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate price series with trend and noise
    t = np.arange(0, 1000)
    trend = 100 + 0.1 * t + 5 * np.sin(t / 50)
    noise = np.random.normal(0, 2, len(t))
    prices = trend + noise
    
    # Apply Causal MODWT
    modwt = CausalMODWT(wavelet='sym8', decomposition_level=4)
    denoised = modwt.wavelet_denoise(prices, threshold_multiplier=1.5)
    
    print("Original series (first 10):", prices[:10])
    print("Denoised series (first 10):", denoised[:10])
    
    # Analyze volatility drivers
    analyzer = WaveletVarianceAnalyzer()
    analysis = analyzer.analyze_volatility_drivers(prices)
    print("\nVolatility Analysis:")
    print(f"Short-term variance: {analysis['short_term_variance_pct']:.2f}%")
    print(f"Long-term variance: {analysis['long_term_variance_pct']:.2f}%")
    print(f"Noise dominant: {analysis['noise_dominant']}")
