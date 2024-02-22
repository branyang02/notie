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

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Tensor, shape [seq_len, d_k]
            K: Tensor, shape [seq_len, d_k]
            V: Tensor, shape [seq_len, d_v]
            mask: Tensor, shape [seq_len, seq_len]
        """
        d_k = Q.size(-1)
        # Compute Attention Scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

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


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, X):
        """
        Args:
            X: Tensor, shape [seq_len, d_model]
        """
        output = self.linear2(self.relu(self.linear1(X)))
        return output


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

        # Initialize Multi-Head Attention
        self.multi_head_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads)

        # Initialize Feed-Forward Network
        self.feed_forward_network = FeedForwardNetwork(d_model, d_ff)

        # Initialize Layer Normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, Z):
        """
        Args:
            X: Tensor, shape [seq_len, d_model]
        """
        # Apply Multi-Head Attention
        multi_head_output = self.multi_head_attention(Z)

        # Apply Layer Normalization
        layer_norm_output1 = self.layer_norm1(Z + multi_head_output)

        # Apply Feed-Forward Network
        feed_forward_output = self.feed_forward_network(layer_norm_output1)

        # Apply Layer Normalization
        layer_norm_output2 = self.layer_norm2(layer_norm_output1 + feed_forward_output)
        return layer_norm_output2


class EncoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, num_encoders):
        super(EncoderOnlyTransformer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.num_encoders = num_encoders

        # Initialize Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Initialize Encoders
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(d_model, n_heads, d_k, d_v, d_ff)
                for _ in range(num_encoders)
            ]
        )

    def forward(self, token_embeddings):
        """
        Args:
            token_embeddings: Tensor, shape [seq_len, d_model]
        """
        # Apply Positional Encoding
        token_embeddings = self.positional_encoding(token_embeddings)

        # Apply Encoders
        for encoder in self.encoders:
            token_embeddings = encoder(token_embeddings)
        return token_embeddings


# Example Usage
d_model = 512  # Dimension of the model (i.e., token embeddings)
seq_length = 32  # Length of the input sequence
num_heads = 8
d_k = d_v = d_model // num_heads
d_ff = 2048  # Dimension of the feed-forward network
num_encoders = 6  # Number of encoders in the transformer

# Arbitrary token embeddings
token_embeddings = torch.randn(seq_length, d_model)
print("Input: ")
print(f"Token Embeddings Shape: {token_embeddings.shape}\n")

# Initialize Encoder-Only Transformer
encoder_only_transformer = EncoderOnlyTransformer(
    d_model, num_heads, d_k, d_v, d_ff, num_encoders
)

# Apply Encoder-Only Transformer
output = encoder_only_transformer(token_embeddings)
print("Output: ")
print(f"Output Shape: {output.shape}")
