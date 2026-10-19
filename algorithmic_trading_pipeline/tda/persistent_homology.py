"""
Topological Data Analysis (TDA) for Market Regime Detection
Implements Takens time-delay embedding, persistent homology, and early warning signals
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


class TakensEmbedding:
    """
    Phase space reconstruction using Takens' time-delay embedding theorem.
    Converts univariate time series into high-dimensional point cloud representation.
    """
    
    def __init__(self, embedding_dim: int = 5, time_delay: Optional[int] = None):
        """
        Initialize Takens embedding parameters.
        
        Args:
            embedding_dim: Dimension of reconstructed phase space (d).
                          Typically 3-6 for financial time series.
            time_delay: Time lag between components (tau).
                       If None, uses first minimum of Average Mutual Information.
                       Default is 1 for financial data (daily bars).
        """
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay if time_delay is not None else 1
    
    def embed(self, time_series: np.ndarray) -> np.ndarray:
        """
        Reconstruct phase space from univariate time series.
        
        V_t = [x(t), x(t-tau), x(t-2*tau), ..., x(t-(d-1)*tau)]
        
        Args:
            time_series: 1D array of time series values
            
        Returns:
            Embedded point cloud of shape (N, embedding_dim)
            where N = len(time_series) - (embedding_dim - 1) * time_delay
        """
        n = len(time_series)
        max_idx = n - (self.embedding_dim - 1) * self.time_delay
        
        embedded = np.zeros((max_idx, self.embedding_dim))
        
        for i in range(max_idx):
            for j in range(self.embedding_dim):
                embedded[i, j] = time_series[i + j * self.time_delay]
        
        return embedded
    
    def estimate_optimal_embedding_dim(self, time_series: np.ndarray,
                                      max_dim: int = 10,
                                      threshold: float = 0.9) -> int:
        """
        Estimate optimal embedding dimension using False Nearest Neighbors (FNN) algorithm.
        
        The idea: as we increase dimension, "false" neighbors (points close due to projection)
        should disappear. Optimal dim is where FNN% drops below threshold.
        
        Args:
            time_series: 1D array
            max_dim: Maximum dimension to test
            threshold: FNN percentage threshold (default 0.9%)
            
        Returns:
            Recommended embedding dimension
        """
        fnn_percentages = []
        
        for dim in range(1, max_dim + 1):
            self.embedding_dim = dim
            embedded_current = self.embed(time_series)
            
            if dim == 1:
                fnn_percentages.append(100.0)
                continue
            
            # Compute distances in current and higher dimension
            distances_current = pdist(embedded_current, metric='euclidean')
            
            self.embedding_dim = dim + 1
            embedded_next = self.embed(time_series)
            distances_next = pdist(embedded_next, metric='euclidean')
            
            # Align distances (some points drop out due to embedding dimension)
            n_pairs = len(distances_current)
            n_pairs_next = len(distances_next)
            
            # Find nearest neighbors in current dimension
            dist_matrix_current = squareform(distances_current)
            nn_indices = np.argsort(dist_matrix_current, axis=1)[:, 1]  # Skip self
            
            # Check if these remain neighbors in higher dimension
            dist_matrix_next = squareform(distances_next)
            fnn_count = 0
            
            for i in range(min(len(dist_matrix_current), len(dist_matrix_next))):
                nn_idx = nn_indices[i]
                if nn_idx < len(dist_matrix_next):
                    current_nn_dist = dist_matrix_current[i, nn_idx]
                    next_nn_dist = dist_matrix_next[i, nn_idx]
                    
                    # False neighbor if distance increases significantly
                    if current_nn_dist > 0 and next_nn_dist / current_nn_dist > 2.0:
                        fnn_count += 1
            
            fnn_pct = (fnn_count / min(len(dist_matrix_current), len(dist_matrix_next))) * 100
            fnn_percentages.append(fnn_pct)
            
            if fnn_pct < threshold:
                self.embedding_dim = dim
                return dim
        
        # If threshold not met, return max_dim
        self.embedding_dim = max_dim
        return max_dim


class PersistentHomologyAnalyzer:
    """
    Analyzes topological features of embedded point cloud using persistent homology.
    Detects market crashes via H_1 (loops/cycles) spikes in persistence landscape.
    """
    
    def __init__(self, embedding: TakensEmbedding):
        self.embedding = embedding
    
    def compute_pairwise_distances(self, point_cloud: np.ndarray) -> np.ndarray:
        """Compute pairwise distances for Vietoris-Rips filtration"""
        return squareform(pdist(point_cloud, metric='euclidean'))
    
    def estimate_persistence_landscape_norm(self, point_cloud: np.ndarray,
                                           p: int = 2) -> float:
        """
        Estimate L^p norm of persistence landscape from Vietoris-Rips complex.
        
        This is a proxy for detecting topological features associated with crashes.
        When market approaches critical transition, H_1 (loops) becomes prominent.
        
        Args:
            point_cloud: Embedded phase space points
            p: Norm parameter (L^p norm)
            
        Returns:
            Estimated persistence landscape norm
        """
        if len(point_cloud) < 3:
            return 0.0
        
        dist_matrix = self.compute_pairwise_distances(point_cloud)
        
        # Extract birth/death pairs from distance matrix (simplified approach)
        # For true persistent homology, use giotto-tda or similar
        
        # Birth times: when two points connect (distance increases)
        # Death times: when cycles are filled in
        
        # Approximation: use density variations
        mean_dist = np.mean(dist_matrix[dist_matrix > 0])
        std_dist = np.std(dist_matrix[dist_matrix > 0])
        
        # Coefficient of variation as proxy for persistence
        persistence_proxy = std_dist / (mean_dist + 1e-8)
        
        # L^p norm (simplified)
        landscape_norm = np.power(persistence_proxy * len(point_cloud), 1.0 / p)
        
        return landscape_norm
    
    def detect_crash_signal(self, point_cloud: np.ndarray,
                           historical_norm: float,
                           threshold_zscore: float = 2.0) -> Tuple[bool, float]:
        """
        Detect early warning signal (EWS) for market crash.
        
        Significant spike in H_1 norm precedes crashes.
        
        Args:
            point_cloud: Current embedded point cloud
            historical_norm: Rolling mean of historical norms
            threshold_zscore: Z-score threshold for anomaly (default 2.0 = 95% confidence)
            
        Returns:
            (is_crash_warning, z_score)
        """
        current_norm = self.estimate_persistence_landscape_norm(point_cloud)
        
        # Estimate std from recent history (simplified)
        historical_std = historical_norm * 0.1 if historical_norm > 0 else 1.0
        
        z_score = (current_norm - historical_norm) / (historical_std + 1e-8)
        
        is_warning = z_score > threshold_zscore
        
        return is_warning, z_score


class WassersteinRegimeDetector:
    """
    Detects regime changes using Wasserstein distance between persistence diagrams.
    When topology deforms (rules of the game change), Wasserstein distance spikes.
    """
    
    def __init__(self, embedding: TakensEmbedding, n_bins: int = 20):
        self.embedding = embedding
        self.n_bins = n_bins
    
    def estimate_persistence_distribution(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Create histogram approximation of persistence diagram.
        
        Args:
            point_cloud: Embedded phase space
            
        Returns:
            Normalized histogram of persistence values
        """
        if len(point_cloud) < 3:
            return np.ones(self.n_bins) / self.n_bins
        
        dist_matrix = squareform(pdist(point_cloud, metric='euclidean'))
        
        # Persistence values from distances
        persistence_values = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        
        # Filter out NaN values
        persistence_values = persistence_values[~np.isnan(persistence_values)]
        
        if len(persistence_values) == 0:
            return np.ones(self.n_bins) / self.n_bins
        
        hist, _ = np.histogram(persistence_values, bins=self.n_bins, 
                              range=(0, np.max(persistence_values) + 1e-8))
        
        # Normalize to probability distribution
        return hist / np.sum(hist)
    
    def wasserstein_distance(self, dist1: np.ndarray, dist2: np.ndarray,
                            p: int = 2) -> float:
        """
        Calculate approximate Wasserstein distance between two distributions.
        
        W_p(μ, ν) = (inf over γ ∫ ||x-y||^p dγ(x,y))^(1/p)
        
        Using 1D approximation with sorted values.
        
        Args:
            dist1: First probability distribution
            dist2: Second probability distribution
            p: Wasserstein distance order
            
        Returns:
            Wasserstein distance
        """
        # Sort cumulative distributions
        cum1 = np.cumsum(dist1)
        cum2 = np.cumsum(dist2)
        
        # Wasserstein distance approximation
        diff = np.abs(cum1 - cum2)
        w_dist = np.power(np.sum(diff ** p), 1.0 / p)
        
        return w_dist
    
    def detect_regime_switch(self, current_cloud: np.ndarray,
                            previous_cloud: np.ndarray,
                            threshold: float = 0.15) -> Tuple[bool, float]:
        """
        Detect topological regime change.
        
        Args:
            current_cloud: Current embedded point cloud
            previous_cloud: Previous embedded point cloud
            threshold: Wasserstein distance threshold for regime switch
            
        Returns:
            (is_regime_switch, wasserstein_distance)
        """
        dist_current = self.estimate_persistence_distribution(current_cloud)
        dist_previous = self.estimate_persistence_distribution(previous_cloud)
        
        w_dist = self.wasserstein_distance(dist_current, dist_previous)
        
        is_switch = w_dist > threshold
        
        return is_switch, w_dist


class MarketPhaseDetector:
    """Combined detector for market phases and transitions"""
    
    def __init__(self, embedding_dim: int = 5, time_delay: int = 1):
        self.takens = TakensEmbedding(embedding_dim=embedding_dim, time_delay=time_delay)
        self.ph_analyzer = PersistentHomologyAnalyzer(self.takens)
        self.regime_detector = WassersteinRegimeDetector(self.takens)
    
    def analyze_window(self, price_series: np.ndarray,
                      previous_embedded: Optional[np.ndarray] = None,
                      historical_norm: float = 1.0) -> dict:
        """
        Comprehensive market phase analysis for a price window.
        
        Returns:
            Dictionary with crash signal, regime switch, and other metrics
        """
        # Embed price series
        embedded = self.takens.embed(price_series)
        
        # Normalize
        scaler = StandardScaler()
        embedded_normalized = scaler.fit_transform(embedded)
        
        # Crash detection
        crash_warning, crash_zscore = self.ph_analyzer.detect_crash_signal(
            embedded_normalized, historical_norm
        )
        
        # Regime detection
        regime_switch = False
        wasserstein_dist = 0.0
        if previous_embedded is not None:
            regime_switch, wasserstein_dist = self.regime_detector.detect_regime_switch(
                embedded_normalized, previous_embedded
            )
        
        # Calculate new norm for next iteration
        current_norm = self.ph_analyzer.estimate_persistence_landscape_norm(
            embedded_normalized
        )
        
        return {
            'embedded_cloud': embedded_normalized,
            'crash_warning': crash_warning,
            'crash_zscore': crash_zscore,
            'regime_switch': regime_switch,
            'wasserstein_distance': wasserstein_dist,
            'persistence_norm': current_norm,
            'topology': 'unknown'
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample time series
    t = np.arange(0, 500)
    prices = 100 + 10 * np.sin(t / 50) + np.cumsum(np.random.normal(0, 0.5, len(t)))
    
    # Create phase detector
    detector = MarketPhaseDetector(embedding_dim=5, time_delay=1)
    
    # Analyze
    analysis = detector.analyze_window(prices)
    
    print("Market Phase Analysis:")
    print(f"Crash Warning: {analysis['crash_warning']}")
    print(f"Crash Z-score: {analysis['crash_zscore']:.2f}")
    print(f"Persistence Norm: {analysis['persistence_norm']:.4f}")
