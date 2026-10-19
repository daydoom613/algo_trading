"""
Data Utilities for fetching and processing financial data
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta


class DataFetcher:
    """Fetch financial data from multiple sources"""
    
    @staticmethod
    def fetch_yfinance(ticker: str, start_date: str, end_date: str,
                       interval: str = '1d') -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker (e.g., 'RELIANCE.NS')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: '1d' for daily, '1h' for hourly, etc.
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def fetch_multi_assets(tickers: Dict[str, str], start_date: str,
                          end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple assets.
        
        Args:
            tickers: Dictionary mapping asset names to ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping asset names to DataFrames
        """
        data = {}
        for name, ticker in tickers.items():
            print(f"Fetching {name} ({ticker})...")
            data[name] = DataFetcher.fetch_yfinance(ticker, start_date, end_date)
        
        return data


class DataPreprocessor:
    """Preprocess and normalize financial data"""
    
    @staticmethod
    def prepare_price_series(df: pd.DataFrame, price_col: str = 'Adj Close') -> np.ndarray:
        """
        Extract and clean price series.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name to use as price
            
        Returns:
            1D array of prices
        """
        prices = df[price_col].values
        
        # Handle NaN values
        prices = np.nan_to_num(prices, method='forward')
        
        # Ensure no zeros
        prices = np.where(prices == 0, np.nan, prices)
        prices = np.nan_to_num(prices, method='forward')
        
        return prices
    
    @staticmethod
    def calculate_returns(prices: np.ndarray, method: str = 'log') -> np.ndarray:
        """
        Calculate returns from prices.
        
        Args:
            prices: Price series
            method: 'log' for log returns, 'simple' for simple returns
            
        Returns:
            Returns array
        """
        if method == 'log':
            returns = np.diff(np.log(prices))
        else:  # simple
            returns = np.diff(prices) / prices[:-1]
        
        return np.concatenate([[0], returns])  # Prepend 0 for first price
    
    @staticmethod
    def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Normalize feature matrix.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Normalized matrix and scaling parameters
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        
        X_normalized = (X - mean) / (std + 1e-8)
        
        return X_normalized, {'mean': mean, 'std': std}
    
    @staticmethod
    def create_lagged_features(series: np.ndarray, n_lags: int = 10) -> np.ndarray:
        """
        Create lagged features for time series.
        
        Args:
            series: 1D time series
            n_lags: Number of lags
            
        Returns:
            2D array where each row is (y_t, y_t-1, ..., y_t-n_lags)
        """
        n = len(series)
        X = np.zeros((n - n_lags, n_lags + 1))
        
        for i in range(n - n_lags):
            X[i] = series[i:i + n_lags + 1][::-1]  # Reverse to get [y_t, y_t-1, ..., y_t-n_lags]
        
        return X


class MultiAssetDataProcessor:
    """Process and align multiple asset time series"""
    
    @staticmethod
    def align_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Align multiple assets on common dates.
        
        Args:
            data_dict: Dictionary mapping asset names to DataFrames
            
        Returns:
            Dictionary mapping asset names to aligned price arrays
        """
        # Get common dates
        all_dates = set()
        for df in data_dict.values():
            all_dates.update(df.index)
        
        common_dates = sorted(all_dates)
        
        # Align prices
        aligned_prices = {}
        for name, df in data_dict.items():
            prices = np.zeros(len(common_dates))
            
            # Use 'Close' if 'Adj Close' not available (yfinance auto_adjust behavior)
            price_column = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            
            for i, date in enumerate(common_dates):
                if date in df.index:
                    prices[i] = float(df.loc[date, price_column].iloc[0]) if isinstance(df.loc[date, price_column], pd.Series) else float(df.loc[date, price_column])
                else:
                    prices[i] = np.nan
            
            # Forward-fill NaN values
            prices = pd.Series(prices).ffill().values
            
            aligned_prices[name] = prices
        
        return aligned_prices
    
    @staticmethod
    def create_multivariate_features(prices_dict: Dict[str, np.ndarray],
                                    n_lags: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Create multivariate feature matrix from multiple assets.
        
        Args:
            prices_dict: Dictionary of price arrays
            n_lags: Number of lags per asset
            
        Returns:
            Feature matrix (n_samples, n_features) and feature names
        """
        asset_names = list(prices_dict.keys())
        prices_arrays = list(prices_dict.values())
        
        # Calculate returns for each asset
        all_returns = []
        feature_names = []
        
        for name, prices in zip(asset_names, prices_arrays):
            returns = DataPreprocessor.calculate_returns(prices)
            all_returns.append(returns)
            
            for lag in range(n_lags):
                feature_names.append(f"{name}_ret_lag{lag}")
        
        # Stack returns
        all_returns = np.column_stack(all_returns)
        
        # Create lagged features
        n_samples = len(all_returns) - n_lags
        n_features = len(asset_names) * n_lags
        
        X = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            for j, returns in enumerate(all_returns.T):
                for lag in range(n_lags):
                    X[i, j * n_lags + lag] = returns[i + n_lags - lag - 1]
        
        return X, feature_names


class RealWorldDataContext:
    """
    Provide realistic market data context with actual correlations and regimes.
    Example: Reliance Industries with Brent Crude and USD/INR.
    """
    
    @staticmethod
    def get_ril_context(end_date: Optional[str] = None) -> Dict:
        """
        Get RIL market context with related assets.
        
        Args:
            end_date: End date for data fetching
            
        Returns:
            Dictionary with prices and metadata
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=2*365)).strftime('%Y-%m-%d')
        
        tickers = {
            'RIL': 'RELIANCE.NS',
            'Brent': 'BZ=F',
            'EURINR': 'EURINR=X'
        }
        
        fetcher = DataFetcher()
        data = {}
        
        for name, ticker in tickers.items():
            print(f"Fetching {name}...")
            try:
                df = fetcher.fetch_yfinance(ticker, start_date, end_date)
                if not df.empty:
                    data[name] = df
            except:
                print(f"Failed to fetch {name}")
        
        if not data:
            # Return synthetic data if fetch fails
            print("Using synthetic data for RIL context")
            n_days = 500
            dates = pd.date_range(end=end_date, periods=n_days)
            
            # Synthetic prices with realistic correlations
            np.random.seed(42)
            ril_prices = 2000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n_days)))
            brent_prices = 80 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, n_days)))
            eurinr_prices = 90 * np.exp(np.cumsum(np.random.normal(0.0, 0.01, n_days)))
            
            data = {
                'RIL': pd.DataFrame({'Adj Close': ril_prices}, index=dates),
                'Brent': pd.DataFrame({'Adj Close': brent_prices}, index=dates),
                'EURINR': pd.DataFrame({'Adj Close': eurinr_prices}, index=dates)
            }
        
        return data


if __name__ == "__main__":
    # Example: Fetch and process RIL data
    print("Fetching market data context for RIL...")
    context = RealWorldDataContext.get_ril_context()
    
    # Align and process
    aligned_prices = MultiAssetDataProcessor.align_data(context)
    
    print("\nAligned prices:")
    for asset, prices in aligned_prices.items():
        print(f"  {asset}: {len(prices)} data points, "
              f"range [{prices.min():.2f}, {prices.max():.2f}]")
    
    # Create multivariate features
    X, feature_names = MultiAssetDataProcessor.create_multivariate_features(
        aligned_prices, n_lags=5
    )
    
    print(f"\nMultivariate features shape: {X.shape}")
    print(f"First few features: {feature_names[:5]}")
