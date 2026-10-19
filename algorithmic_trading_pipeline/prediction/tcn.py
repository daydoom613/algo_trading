"""
Temporal Convolutional Network (TCN) for Financial Time Series Forecasting
Implements dilated causal convolutions with attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution that ensures output at time t depends only on
    inputs from time t and earlier (no future information).
    
    Implemented via left-padding the input by (kernel_size - 1) * dilation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, dropout: float = 0.2):
        super(CausalConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Causal padding amount (left side only)
        self.causal_padding = (kernel_size - 1) * dilation
        
        # Actual convolution
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0, bias=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal padding.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len)
        """
        # Left-pad input with zeros (causal constraint)
        x_padded = F.pad(x, (self.causal_padding, 0))
        
        # Apply convolution
        out = self.conv(x_padded)
        
        # Apply dropout
        out = self.dropout(out)
        
        return out


class TemporalBlock(nn.Module):
    """
    Building block for TCN with residual connection and normalization.
    """
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        self.causal_conv1 = CausalConv1d(
            n_inputs, n_outputs, kernel_size, dilation=dilation, dropout=dropout
        )
        self.causal_conv2 = CausalConv1d(
            n_outputs, n_outputs, kernel_size, dilation=dilation, dropout=dropout
        )
        
        self.net = nn.Sequential(
            self.causal_conv1,
            nn.ReLU(),
            self.causal_conv2,
            nn.ReLU()
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with small values to stabilize training"""
        self.causal_conv1.conv.weight.data.normal_(0, 0.01)
        self.causal_conv2.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (batch_size, n_inputs, seq_len)
            
        Returns:
            Output tensor (batch_size, n_outputs, seq_len)
        """
        out = self.net(x)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class AttentionLayer(nn.Module):
    """
    Multi-head self-attention to weight importance of different time steps.
    Especially useful for giving more weight to periods with regime shifts.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super(AttentionLayer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)
    
    def forward(self, values: torch.Tensor, keys: torch.Tensor,
                query: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            values, keys, query: (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Attention output and attention weights
        """
        batch_size = query.shape[0]
        
        # Transform
        Q = self.query(query)
        K = self.key(keys)
        V = self.value(values)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(energy, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.d_model)
        
        out = self.fc_out(out)
        
        return out, attention_weights


class TemporalConvolutionalNetwork(nn.Module):
    """
    TCN with dilated causal convolutions and optional attention.
    
    Architecture:
    - Input -> Temporal Blocks (exponentially increasing dilation) -> Attention -> Output Head
    """
    
    def __init__(self, input_size: int, output_size: int = 1,
                 num_channels: int = 32, kernel_size: int = 5,
                 dropout: float = 0.2, num_levels: int = 4,
                 use_attention: bool = True, num_heads: int = 4):
        super(TemporalConvolutionalNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.use_attention = use_attention
        self.receptive_field = self._calculate_receptive_field(kernel_size, num_levels)
        
        # Build temporal blocks with exponentially increasing dilation
        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels
            out_channels = num_channels
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                dilation=dilation_size, dropout=dropout
            ))
        
        self.temporal_blocks = nn.Sequential(*layers)
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(num_channels, num_heads=num_heads, dropout=dropout)
        
        # Output head
        self.fc = nn.Linear(num_channels, output_size)
    
    def _calculate_receptive_field(self, kernel_size: int, num_levels: int) -> int:
        """
        Calculate receptive field: how far back the network can see.
        
        R = 1 + sum((k-1) * d_i) for i in 1..L
        where k is kernel size, L is num levels, d_i is dilation at level i
        """
        receptive_field = 1
        for i in range(num_levels):
            dilation = 2 ** i
            receptive_field += (kernel_size - 1) * dilation
        
        return receptive_field
    
    def forward(self, x: torch.Tensor, regime_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through TCN.
        
        Args:
            x: Input tensor (batch_size, input_size, seq_len)
            regime_weights: Optional weights for different time steps (for attention masking)
            
        Returns:
            Predictions (batch_size, seq_len, output_size) and attention weights
        """
        # Temporal convolutions
        tcn_out = self.temporal_blocks(x)  # (batch_size, num_channels, seq_len)
        
        attention_weights = None
        
        if self.use_attention:
            # Transpose for attention layer
            tcn_out_t = tcn_out.transpose(1, 2)  # (batch_size, seq_len, num_channels)
            
            # Apply attention
            attended_out, attention_weights = self.attention(
                tcn_out_t, tcn_out_t, tcn_out_t, mask=None
            )
            
            # Use regime weights if provided
            if regime_weights is not None:
                attended_out = attended_out * regime_weights.unsqueeze(-1)
            
            tcn_out_t = attended_out
        else:
            tcn_out_t = tcn_out.transpose(1, 2)  # (batch_size, seq_len, num_channels)
        
        # Output head
        predictions = self.fc(tcn_out_t)  # (batch_size, seq_len, output_size)
        
        return predictions, attention_weights
    
    def get_receptive_field(self) -> int:
        """Get receptive field size"""
        return self.receptive_field


class TCNPredictor:
    """High-level wrapper for TCN prediction"""
    
    def __init__(self, input_size: int, output_size: int = 1,
                 num_channels: int = 32, kernel_size: int = 5,
                 dropout: float = 0.2, num_levels: int = 4,
                 use_attention: bool = True, learning_rate: float = 0.001,
                 device: str = 'cpu'):
        
        self.device = device
        self.model = TemporalConvolutionalNetwork(
            input_size, output_size, num_channels, kernel_size,
            dropout, num_levels, use_attention
        ).to(device)
        
        # Initialize weights with Xavier uniform for better convergence
        self._init_weights()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.SmoothL1Loss()  # More stable than MSE
    
    def _init_weights(self):
        """Initialize network weights with Xavier uniform initialization"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Single training step.
        
        Args:
            X: Input features (batch_size, seq_len, input_size)
            y: Target values (batch_size, seq_len, output_size) or (batch_size, output_size)
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Handle NaN values in input
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Clip extreme values
        X = np.clip(X, -1e4, 1e4)
        y = np.clip(y, -1e4, 1e4)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        
        # Transpose if needed
        if X_tensor.dim() == 3:
            X_tensor = X_tensor.transpose(1, 2)  # (batch, input_size, seq_len)
        
        # Forward pass
        predictions, _ = self.model(X_tensor)
        
        # Ensure shapes match
        if y_tensor.dim() == 2:
            y_tensor = y_tensor.unsqueeze(1)
        
        # Clip predictions to prevent divergence
        predictions = torch.clamp(predictions, -1e4, 1e4)
        
        # Compute loss
        loss = self.criterion(predictions, y_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        loss_val = loss.item()
        # Return safe loss value
        return 0.0 if np.isnan(loss_val) or np.isinf(loss_val) else loss_val
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (batch_size, seq_len, input_size) or (seq_len, input_size)
            
        Returns:
            Predictions (batch_size, seq_len, output_size)
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            
            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(0)  # Add batch dimension
            
            # Transpose for model
            X_tensor = X_tensor.transpose(1, 2)  # (batch, input_size, seq_len)
            
            predictions, _ = self.model(X_tensor)
            
            return predictions.cpu().numpy()


if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    tcn = TCNPredictor(input_size=3, output_size=1, num_channels=32,
                      kernel_size=5, num_levels=4, use_attention=True,
                      device=device)
    
    print(f"Model receptive field: {tcn.model.get_receptive_field()} steps")
    
    # Dummy data
    X_train = np.random.randn(32, 100, 3)  # (batch, seq_len, features)
    y_train = np.random.randn(32, 100, 1)  # (batch, seq_len, target)
    
    # Training
    for epoch in range(5):
        loss = tcn.train_step(X_train, y_train)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    
    # Prediction
    X_test = np.random.randn(1, 100, 3)
    predictions = tcn.predict(X_test)
    print(f"Prediction shape: {predictions.shape}")
