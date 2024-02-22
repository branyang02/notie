import torch
import math
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        """
        Args:
            Q: Tensor, shape [seq_len, d_k]
            K: Tensor, shape [seq_len, d_k]
            V: Tensor, shape [seq_len, d_v]
        """
        d_k = Q.size(-1)

        # Compute Attention Scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Compute Attention Weights
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Compute Attention Output
        attention_output = torch.matmul(attention_weights, V)
        return attention_output


# Example Usage
d_k = d_v = 64  # Dimension of the key and value
seq_length = 32  # Length of the input sequence

# Initialize Q, K, V
Q = torch.randn(seq_length, d_k)
K = torch.randn(seq_length, d_k)
V = torch.randn(seq_length, d_v)
print("Input: ")
print(f"Q Shape: {Q.shape} \nK Shape: {K.shape} \nV Shape: {V.shape}\n")

# Initialize Scaled Dot-Product Attention
scaled_dot_product_attention = ScaledDotProductAttention()

# Apply Scaled Dot-Product Attention
attention_output = scaled_dot_product_attention(Q, K, V)
print("Output: ")
print(f"Scaled Dot-Product Attention Output Shape: {attention_output.shape}")
