import torch
import math
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Tensor, shape [seq_len, d_k]
            K: Tensor, shape [seq_len, d_k]
            V: Tensor, shape [seq_len, d_v]
            mask: Tensor, shape [seq_len, seq_len]
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            print("Masked Scores: \n", scores)
        else:
            print("Unmasked Scores: \n", scores)

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        if mask is not None:
            print("Masked Attention Weights: \n", attention_weights)
        else:
            print("Unmasked Attention Weights: \n", attention_weights)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output


# Example Usage
d_k = d_v = 8  # Dimension of the key and value
seq_length = 5  # Length of the input sequence

# Initialize Q, K, V
Q = torch.randn(seq_length, d_k)
K = torch.randn(seq_length, d_k)
V = torch.randn(seq_length, d_v)
print("Input: Query, Key, Value")
print(f"Q Shape: {Q.shape} \nK Shape: {K.shape} \nV Shape: {V.shape}\n")

# Initialize Scaled Dot-Product Attention
scaled_dot_product_attention = ScaledDotProductAttention()

# Apply unmasked Scaled Dot-Product Attention
attention_output = scaled_dot_product_attention(Q, K, V)

# Apply Scaled Dot-Product Attention with Mask
print("\nApply Scaled Dot-Product Attention with Mask")

# Create a mask where only previous positions are attended to
mask = ~torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
print("Mask: \n", mask)
attention_output = scaled_dot_product_attention(Q, K, V, mask)


print("\nOutput: ")
print(f"Scaled Dot-Product Attention Output Shape: {attention_output.shape}")
