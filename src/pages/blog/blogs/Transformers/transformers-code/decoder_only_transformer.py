import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pprint import pprint


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


class MaskedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MaskedMultiHeadAttention, self).__init__(d_model, d_k, d_v, n_heads)

    def forward(self, Z):
        """
        Args:
            Z: Tensor, shape [seq_len, d_model]
        """
        seq_length = Z.size(0)
        mask = ~torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()

        results = []
        for i in range(self.n_heads):
            Q, K, V = self.attentions[i](Z)
            attention_output = self.scaled_dot_product_attentions[i](Q, K, V, mask)
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


class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text):
        """Encode a text string to a list of token IDs."""
        tokens = text.split()
        token_ids = [self.vocab.get(token) for token in tokens]
        return token_ids

    def decode(self, token_ids):
        """Decode a list of token IDs or a PyTorch tensor of token IDs to a text string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens = [self.inverse_vocab.get(id) for id in token_ids]
        text = " ".join(tokens)
        return text


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

        # Masked Multi-Head Attention
        self.masked_multi_head_attention = MaskedMultiHeadAttention(
            d_model, d_k, d_v, n_heads
        )

        # Multi-Head Attention
        self.multi_head_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads)

        # Add & Norm Layers
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        # Feed Forward Network
        self.feed_forward_network = FeedForwardNetwork(d_model, d_ff)

    def forward(self, Z):
        """
        Args:
            Z: Tensor, shape [seq_len, d_model]
        """
        # Masked Multi-Head Attention
        masked_multi_head_attention_output = self.masked_multi_head_attention(Z)

        # Add & Norm
        layer_norm_output1 = self.layer_norm1(Z + masked_multi_head_attention_output)

        # Multi-Head Attention
        multi_head_attention_output = self.multi_head_attention(layer_norm_output1)
        # Add & Norm
        layer_norm_output2 = self.layer_norm2(
            layer_norm_output1 + multi_head_attention_output
        )

        # Feed Forward Network
        feed_forward_output = self.feed_forward_network(layer_norm_output2)

        # Add & Norm
        layer_norm_output3 = self.layer_norm3(layer_norm_output2 + feed_forward_output)

        return layer_norm_output3


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, vocab_size, num_decoders):
        super(DecoderOnlyTransformer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.num_decoders = num_decoders

        # Initialize Token Embedding Layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Initialize Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, n_heads, d_k, d_v, d_ff)
                for _ in range(num_decoders)
            ]
        )

        # Final Linear Layer
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        """
        Args:
            token_ids: Tensor, shape [input_seq_len]

        Returns:
            next_token: Tensor, shape [1]
        """
        # Token Embedding
        token_embeddings = self.token_embedding(token_ids)

        # Positional Encoding
        positional_encoding_output = self.positional_encoding(token_embeddings)

        # Decoder Blocks
        for decoder_block in self.decoder_blocks:
            positional_encoding_output = decoder_block(positional_encoding_output)

        # Final Linear Layer
        logits = self.linear(positional_encoding_output)

        # Softmax Layer to get probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Sample the next token from the probability distribution
        next_tokens = torch.multinomial(probabilities, num_samples=1)

        next_token = next_tokens[-1]

        return next_token

    def inference(self, token_ids, max_seq_len=512):
        """
        Args:
            token_ids: Tensor, shape [input_seq_len]
            max_seq_len: int, maximum sequence length

        Returns:
            output: Tensor, shape [output_seq_len]
        """

        generated_token_ids = token_ids.tolist()
        for _ in range(max_seq_len):
            next_token = self.forward(torch.tensor(generated_token_ids))
            generated_token_ids.append(next_token.item())
        return torch.tensor(generated_token_ids)


# Define a simple vocabulary
vocabulary = {
    "hello": 0,
    "world": 1,
    "goodbye": 2,
    "transformer": 3,
    "language": 4,
    "model": 5,
    "learning": 6,
    "deep": 7,
    "neural": 8,
    "network": 9,
    "data": 10,
    "science": 11,
    "machine": 12,
    "artificial": 13,
    "intelligence": 14,
    "supervised": 15,
    "unsupervised": 16,
    "reinforcement": 17,
    "mathematics": 18,
    "statistics": 19,
}

print("Toy Vocabulary: ")
pprint(vocabulary)
print("-----------------------------------")
# Initialize the SimpleTokenizer with the vocabulary
tokenizer = SimpleTokenizer(vocabulary)

# Example Usage
d_model = 512
num_heads = 8
d_k = d_v = d_model // num_heads
d_ff = 2048
num_decoders = 6
input_seq_len = 3  # Length of the input sequence
max_seq_len = 10  # Maximum generated sequence length
vocab_size = len(vocabulary)

# Initialize Decoder-Only Transformer
decoder_only_transformer = DecoderOnlyTransformer(
    d_model,
    num_heads,
    d_k,
    d_v,
    d_ff,
    vocab_size,
    num_decoders,
)


# Apply Decoder-Only Transformer
token_ids = torch.randint(0, vocab_size, (input_seq_len,))
print("String input: ", tokenizer.decode(token_ids))
print("Input Tokens: ", token_ids)

output_tokens = decoder_only_transformer.inference(token_ids, max_seq_len=max_seq_len)
print("Output Tokens: ", output_tokens)
print("String output: ", tokenizer.decode(output_tokens))
