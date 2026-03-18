import torch

@torch.no_grad()
def run(
    query_or_key: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to query or key tensor.
    
    The rotation is applied using the formula:
    - Split features into pairs (x1, x2)
    - Rotate: (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
    
    Args:
        query_or_key: Input tensor of shape (batch, seq_len, num_heads, head_dim)
        freqs_cos: Cosine frequencies of shape (seq_len, head_dim)
        freqs_sin: Sine frequencies of shape (seq_len, head_dim)
        
    Returns:
        Rotated tensor of same shape as input
    """
    # Input shape: (batch, seq_len, num_heads, head_dim)
    # freqs shape: (seq_len, head_dim)
    
    # Reshape frequencies to broadcast correctly
    # freqs: (seq_len, head_dim) -> (1, seq_len, 1, head_dim)
    freqs_cos_expanded = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin_expanded = freqs_sin.unsqueeze(0).unsqueeze(2)
    
    # Split the head_dim into pairs for rotation
    # This is the complex number rotation in real space
    # query_or_key: (batch, seq_len, num_heads, head_dim)
    # Split into: (batch, seq_len, num_heads, head_dim // 2, 2)
    x_shape = query_or_key.shape
    x_reshaped = query_or_key.float().reshape(
        x_shape[0], x_shape[1], x_shape[2], -1, 2
    )
    
    # Split frequencies similarly
    freqs_cos_reshaped = freqs_cos_expanded.float().reshape(
        freqs_cos_expanded.shape[0], freqs_cos_expanded.shape[1], freqs_cos_expanded.shape[2], -1, 2
    )
    freqs_sin_reshaped = freqs_sin_expanded.float().reshape(
        freqs_sin_expanded.shape[0], freqs_sin_expanded.shape[1], freqs_sin_expanded.shape[2], -1, 2
    )
    
    # Extract real and imaginary parts
    x1 = x_reshaped[..., 0]  # (batch, seq_len, num_heads, head_dim // 2)
    x2 = x_reshaped[..., 1]  # (batch, seq_len, num_heads, head_dim // 2)
    
    cos1 = freqs_cos_reshaped[..., 0]  # (1, seq_len, 1, head_dim // 2)
    cos2 = freqs_cos_reshaped[..., 1]  # (1, seq_len, 1, head_dim // 2)
    sin1 = freqs_sin_reshaped[..., 0]  # (1, seq_len, 1, head_dim // 2)
    sin2 = freqs_sin_reshaped[..., 1]  # (1, seq_len, 1, head_dim // 2)
    
    # Apply rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    # This is the complex multiplication: (x1 + ix2) * (cos + isin)
    out1 = x1 * cos1 - x2 * sin1
    out2 = x1 * sin2 + x2 * cos2
    
    # Stack back together
    output = torch.stack([out1, out2], dim=-1)
    
    # Reshape back to original shape
    output = output.reshape(x_shape)
    
    # Convert back to original dtype
    output = output.to(torch.bfloat16)
    
    return output
