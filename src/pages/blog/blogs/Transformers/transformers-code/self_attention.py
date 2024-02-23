import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

        # Initialize Query, Key, Value Weight Matrices
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)

    def forward(self, Z):
        """
        Args:
            Z: Tensor, shape [seq_len, d_model]
        """
        # Compute Q, K, V
        Q = self.W_Q(Z)
        K = self.W_K(Z)
        V = self.W_V(Z)
        return Q, K, V


# Example Usage
d_model = 512  # Dimension of the model (i.e., token embeddings)
d_k = d_v = 64  # Dimension of the key and value
seq_length = 32  # Length of the input sequence

# Initialize Token Embeddings
Z = torch.randn(seq_length, d_model)
print("Input: ")
print(f"Z Shape: {Z.shape}\n")

# Initialize Self-Attention
self_attention = SelfAttention(d_model, d_k, d_v)

# Apply Self-Attention
Q, K, V = self_attention(Z)
print("Output: ")
print(f"Q Shape: {Q.shape} \nK Shape: {K.shape} \nV Shape: {V.shape}")
