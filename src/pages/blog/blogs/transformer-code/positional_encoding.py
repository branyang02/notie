import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, token_embeddings):
        """
        Args:
            token_embeddings: Tensor, shape [seq_len, d_model]
        """
        seq_len, d_model = token_embeddings.size()

        # Generate positional encoding dynamically based on seq_len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(seq_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        # Add positional encoding to token embedding
        token_embeddings = token_embeddings + encoding.to(token_embeddings.device)
        return token_embeddings


# Example Usage
d_model = 512  # Dimension of the model (i.e., token embeddings)
seq_length = 32  # Length of the input sequence

# Adjust token_embeddings to have a shape [seq_length, d_model]
token_embeddings = torch.randn(seq_length, d_model)
print("Input: ")
print(f"Token Embeddings Shape: {token_embeddings.shape}\n")

# Initialize Positional Encoding
pos_encoder = PositionalEncoding(d_model)

# Apply Positional Encoding to Token Embeddings
token_embeddings_with_pos = pos_encoder(token_embeddings)
print("Output: ")
print(
    f"Token Embeddings with Positional Encoding Shape: {token_embeddings_with_pos.shape}"
)
