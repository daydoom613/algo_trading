"""
Backtesting Engine using Vectorized Computations
Implements Walk-Forward Optimization for robust parameter validation
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """Order types"""
    BUY = 1
    SELL = -1
    CLOSE = 0


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005   # 0.05% slippage
    max_leverage: float = 1.0   # No leverage by default
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.03  # 3% take profit
    use_atr_stops: bool = True  # Use ATR-based stops if available
    atr_multiplier: float = 2.0  # ATR multiplier for stops


class VectorizedBacktest:
    """
    Vectorized backtesting engine for high-performance strategy evaluation.
    Processes multiple strategy variations and parameter combinations efficiently.
    """
    
    def __init__(self, prices: np.ndarray, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine.
        
        Args:
            prices: 1D array of price data
            config: Backtesting configuration
        """
        self.prices = prices
        self.n_periods = len(prices)
        self.config = config or BacktestConfig()
        
        self.trades = []
        self.equity_curve = np.zeros(self.n_periods)
        self.position = 0
        self.cash = self.config.initial_capital
    
    def generate_signals(self, signal_func: Callable) -> np.ndarray:
        """
        Generate trading signals using a vectorized function.
        
        Args:
            signal_func: Function that takes prices and returns signal array
                        (1 for buy, -1 for sell, 0 for hold)
        
        Returns:
            Signal array
        """
        return signal_func(self.prices)
    
    def backtest_strategy(self, signals: np.ndarray, atr_values: Optional[np.ndarray] = None) -> Dict:
        """
        Execute backtest on given signals with stop-loss and take-profit.
        
        Args:
            signals: Array of trading signals (1, -1, 0)
            atr_values: Optional ATR values for dynamic stops
            
        Returns:
            Dictionary with performance metrics
        """
        # Initialize tracking arrays
        equity = np.zeros(self.n_periods)
        positions = np.zeros(self.n_periods)
        trades_list = []
        
        current_equity = self.config.initial_capital
        current_position = 0
        entry_price = 0
        entry_time = 0
        stop_loss = 0
        take_profit = 0
        
        for t in range(self.n_periods):
            # Check for stop-loss or take-profit exit
            exit_signal = False
            exit_price = self.prices[t]
            
            if current_position != 0:
                # Check stop-loss
                if current_position > 0 and self.prices[t] <= stop_loss:
                    exit_signal = True
                    exit_price = stop_loss * (1 - self.config.slippage)
                elif current_position < 0 and self.prices[t] >= stop_loss:
                    exit_signal = True
                    exit_price = stop_loss * (1 + self.config.slippage)
                
                # Check take-profit
                if not exit_signal:
                    if current_position > 0 and self.prices[t] >= take_profit:
                        exit_signal = True
                        exit_price = take_profit * (1 - self.config.slippage)
                    elif current_position < 0 and self.prices[t] <= take_profit:
                        exit_signal = True
                        exit_price = take_profit * (1 + self.config.slippage)
            
            # Process signal-based exits/entries
            signal = signals[t]
            
            if exit_signal or (signal != 0 and signal != current_position):
                # Close existing position
                if current_position != 0:
                    if not exit_signal:
                        exit_price = self.prices[t] * (1 - self.config.slippage if current_position > 0 else 1 + self.config.slippage)
                    
                    pnl = (exit_price - entry_price) * current_position
                    current_equity += pnl - (abs(current_position) * 2 * self.config.commission * self.prices[t])
                    
                    trades_list.append({
                        'entry_time': entry_time,
                        'exit_time': t,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'size': current_position,
                        'pnl': pnl,
                        'exit_reason': 'stop_loss' if exit_signal and current_position * (self.prices[t] - entry_price) < 0 else 'take_profit' if exit_signal else 'signal'
                    })
                
                # Open new position on signal (if not from stop/TP)
                if signal != 0 and not exit_signal:
                    entry_price = self.prices[t] * (1 + self.config.slippage if signal > 0 else 1 - self.config.slippage)
                    entry_time = t
                    current_position = signal
                    
                    # Set stops
                    if self.config.use_atr_stops and atr_values is not None:
                        stop_dist = atr_values[t] * self.config.atr_multiplier
                    else:
                        stop_dist = entry_price * self.config.stop_loss_pct
                    
                    if signal > 0:
                        stop_loss = entry_price - stop_dist
                        take_profit = entry_price + (entry_price * self.config.take_profit_pct)
                    else:
                        stop_loss = entry_price + stop_dist
                        take_profit = entry_price - (entry_price * self.config.take_profit_pct)
                else:
                    current_position = 0
            
            # Mark-to-market unrealized P&L
            if current_position != 0:
                unrealized_pnl = (self.prices[t] - entry_price) * current_position
            else:
                unrealized_pnl = 0
            
            current_equity_marked = current_equity + unrealized_pnl
            
            equity[t] = current_equity_marked
            positions[t] = current_position
        
        # Calculate performance metrics
        returns = np.diff(equity) / (equity[:-1] + 1e-8)
        returns = np.concatenate([[0], returns])
        returns = np.nan_to_num(returns, nan=0.0)
        
        total_return = (equity[-1] - self.config.initial_capital) / self.config.initial_capital
        annual_return = (1 + total_return) ** (252 / self.n_periods) - 1 if self.n_periods > 0 else 0
        
        # Sharpe ratio (252 trading days)
        excess_returns = returns - 0.02 / 252  # Assume 2% risk-free rate
        sharpe = np.sqrt(252) * np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
        sharpe = 0 if np.isnan(sharpe) else sharpe
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (cummax - equity) / (cummax + 1e-8)
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Win rate and profit factor
        if len(trades_list) > 0:
            wins = sum(1 for t in trades_list if t['pnl'] > 0)
            win_rate = wins / len(trades_list)
            total_wins = sum(t['pnl'] for t in trades_list if t['pnl'] > 0)
            total_losses = abs(sum(t['pnl'] for t in trades_list if t['pnl'] < 0))
            profit_factor = total_wins / (total_losses + 1e-8) if total_losses > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            'equity': equity,
            'positions': positions,
            'returns': returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(trades_list),
            'trades': trades_list,
            'final_equity': equity[-1]
        }
    
    def vectorized_backtest_sweep(self, signal_func: Callable,
                                 param_ranges: Dict[str, List]) -> List[Dict]:
        """
        Sweep through multiple parameter combinations (grid search).
        Useful for strategy optimization.
        
        Args:
            signal_func: Function that generates signals
            param_ranges: Dictionary mapping param names to ranges
            
        Returns:
            List of results for each parameter combination
        """
        results = []
        
        # Generate all parameter combinations
        import itertools
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            
            # Generate signals with current parameters
            signals = signal_func(self.prices, **params)
            
            # Run backtest
            result = self.backtest_strategy(signals)
            result['params'] = params
            
            results.append(result)
        
        return results


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization for robust parameter validation.
    
    Process:
    1. Divide data into overlapping in-sample (training) and out-of-sample (test) windows
    2. Optimize parameters on in-sample data
    3. Evaluate on out-of-sample data
    4. Repeat with sliding window
    
    This ensures parameters are robust and not just lucky on historical data.
    """
    
    def __init__(self, prices: np.ndarray, config: Optional[BacktestConfig] = None):
        self.prices = prices
        self.config = config or BacktestConfig()
    
    def walk_forward_analysis(self, signal_func: Callable,
                             param_ranges: Dict[str, List],
                             in_sample_size: int = 1000,
                             out_of_sample_size: int = 250,
                             step_size: int = 250) -> Dict:
        """
        Perform walk-forward optimization and analysis.
        
        Args:
            signal_func: Function generating signals
            param_ranges: Parameter ranges for optimization
            in_sample_size: Length of training window
            out_of_sample_size: Length of testing window
            step_size: How many periods to advance window
            
        Returns:
            Dictionary with optimization results and out-of-sample performance
        """
        n = len(self.prices)
        windows = []
        in_sample_results = []
        out_sample_results = []
        
        # Create overlapping windows
        start = 0
        while start + in_sample_size + out_of_sample_size <= n:
            in_sample_end = start + in_sample_size
            out_sample_end = in_sample_end + out_of_sample_size
            
            in_sample_prices = self.prices[start:in_sample_end]
            out_sample_prices = self.prices[in_sample_end:out_sample_end]
            
            windows.append({
                'start': start,
                'in_sample_end': in_sample_end,
                'out_sample_end': out_sample_end
            })
            
            # Optimize on in-sample data
            backtest_in = VectorizedBacktest(in_sample_prices, self.config)
            results_in = backtest_in.vectorized_backtest_sweep(signal_func, param_ranges)
            
            # Find best parameters (by Sharpe ratio)
            best_result = max(results_in, key=lambda r: r['sharpe_ratio'])
            best_params = best_result['params']
            
            in_sample_results.append({
                'window': len(in_sample_results),
                'best_params': best_params,
                'in_sample_sharpe': best_result['sharpe_ratio'],
                'in_sample_return': best_result['total_return']
            })
            
            # Test on out-of-sample data
            backtest_out = VectorizedBacktest(out_sample_prices, self.config)
            signals_out = signal_func(out_sample_prices, **best_params)
            result_out = backtest_out.backtest_strategy(signals_out)
            
            out_sample_results.append({
                'window': len(out_sample_results),
                'out_sample_sharpe': result_out['sharpe_ratio'],
                'out_sample_return': result_out['total_return'],
                'max_drawdown': result_out['max_drawdown'],
                'equity': result_out['equity']
            })
            
            # Advance window
            start += step_size
        
        # Summary statistics
        in_sample_sharpes = [r['in_sample_sharpe'] for r in in_sample_results]
        out_sample_sharpes = [r['out_sample_sharpe'] for r in out_sample_results]
        out_sample_returns = [r['out_sample_return'] for r in out_sample_results]
        
        return {
            'windows': windows,
            'in_sample_results': in_sample_results,
            'out_sample_results': out_sample_results,
            'avg_in_sample_sharpe': np.mean(in_sample_sharpes),
            'avg_out_sample_sharpe': np.mean(out_sample_sharpes),
            'avg_out_sample_return': np.mean(out_sample_returns),
            'out_sample_sharpe_std': np.std(out_sample_sharpes),
            'out_sample_return_std': np.std(out_sample_returns),
            'parameter_stability': self._analyze_parameter_stability(in_sample_results)
        }
    
    def _analyze_parameter_stability(self, results: List[Dict]) -> Dict:
        """
        Analyze stability of optimal parameters across windows.
        
        Returns:
            Dictionary with stability metrics
        """
        if len(results) == 0:
            return {}
        
        # Collect all parameter values across windows
        param_names = list(results[0]['best_params'].keys())
        param_history = {name: [] for name in param_names}
        
        for result in results:
            for name, value in result['best_params'].items():
                param_history[name].append(value)
        
        # Calculate coefficient of variation for each parameter
        stability = {}
        for name, values in param_history.items():
            if isinstance(values[0], (int, float)):
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / (abs(mean_val) + 1e-8)  # Coefficient of variation
                stability[name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv,
                    'stable': cv < 0.2  # Stable if CV < 20%
                }
        
        return stability


class MetricsCalculator:
    """Calculate detailed performance metrics"""
    
    @staticmethod
    def calculate_all_metrics(equity_curve: np.ndarray,
                             returns: np.ndarray,
                             trades: List[Dict],
                             risk_free_rate: float = 0.02) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: Equity curve array
            returns: Returns array
            trades: List of trade records
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of metrics
        """
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        periods = len(equity_curve)
        annual_return = (1 + total_return) ** (252 / periods) - 1
        
        # Sharpe ratio
        excess_returns = returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
        
        # Sortino ratio (only penalize downside)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = np.sqrt(252) * np.mean(excess_returns) / (downside_std + 1e-8)
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (cummax - equity_curve) / cummax
        max_drawdown = np.max(drawdown)
        
        # Calmar ratio
        calmar = annual_return / (max_drawdown + 1e-8)
        
        # Win rate and profit factor
        if len(trades) > 0:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = wins / len(trades)
            
            total_wins = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            total_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            profit_factor = total_wins / (total_losses + 1e-8)
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(trades)
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample price data
    returns = np.random.normal(0.0005, 0.02, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Simple signal function: mean reversion
    def mean_reversion_signal(price, window=20, threshold=1.5):
        ma = pd.Series(price).rolling(window).mean().values
        std = pd.Series(price).rolling(window).std().values
        z_score = (price - ma) / (std + 1e-8)
        
        signals = np.zeros(len(price))
        signals[z_score < -threshold] = 1   # Buy oversold
        signals[z_score > threshold] = -1   # Sell overbought
        return signals
    
    # Backtest configuration
    config = BacktestConfig(initial_capital=100000, commission=0.001)
    
    # Single backtest
    backtest = VectorizedBacktest(prices, config)
    signals = backtest.generate_signals(mean_reversion_signal)
    result = backtest.backtest_strategy(signals)
    
    print("Single Backtest Results:")
    print(f"  Total Return: {result['total_return']:.2%}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"  Win Rate: {result['win_rate']:.2%}")
    
    # Walk-forward optimization
    print("\nWalk-Forward Optimization:")
    wfo = WalkForwardOptimizer(prices, config)
    wfo_results = wfo.walk_forward_analysis(
        mean_reversion_signal,
        {'window': [15, 20, 25], 'threshold': [1.0, 1.5, 2.0]},
        in_sample_size=500, out_of_sample_size=200, step_size=100
    )
    
    print(f"  Avg In-Sample Sharpe: {wfo_results['avg_in_sample_sharpe']:.2f}")
    print(f"  Avg Out-Sample Sharpe: {wfo_results['avg_out_sample_sharpe']:.2f}")
    print(f"  Out-Sample Return: {wfo_results['avg_out_sample_return']:.2%}")
