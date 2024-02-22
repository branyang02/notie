import torch
import torch.nn as nn
import math


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
        # Compute Attention Scores
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        # Compute the Weighted Sum
        attention_output = torch.matmul(attention_weights, V)
        return attention_output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_O = nn.Linear(n_heads * d_v, d_model)
        self.attentions = nn.ModuleList(
            [SelfAttention(d_model, d_k, d_v) for _ in range(n_heads)]
        )
        self.scaled_dot_product_attentions = nn.ModuleList(
            [ScaledDotProductAttention() for _ in range(n_heads)]
        )

    def forward(self, Z):
        """
        Args:
            Z: Tensor, shape [seq_len, d_model]
        """
        results = []
        for i in range(self.n_heads):
            Q, K, V = self.attentions[i](Z)
            attention_output = self.scaled_dot_product_attentions[i](Q, K, V)
            results.append(attention_output)
        # Concatenate the results
        results = torch.cat(results, dim=-1)
        # Apply Linear Transformation
        multi_head_output = self.W_O(results)
        return multi_head_output


# Usage
d_model = 512
seq_length = 32
h = 8
d_k = d_v = d_model // h

# Create a random tensor as input
Z = torch.rand(seq_length, d_model)
print("Input: ")
print(f"Z Shape: {Z.shape}\n")

# Initialize Multi-Head Attention
multi_head_attention = MultiHeadAttention(d_model, d_k, d_v, h)

# Forward Pass
output = multi_head_attention(Z)
print("Output: ")
print(f"Output Shape: {output.shape}")
