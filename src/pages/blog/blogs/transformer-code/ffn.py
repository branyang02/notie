import torch
import torch.nn as nn


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


# Example Usage
d_model = 512  # Dimension of the model (i.e., token embeddings)
d_ff = 2048  # Dimension of the feed-forward network
seq_length = 32  # Length of the input sequence

# Initialize Token Embeddings
X = torch.randn(seq_length, d_model)
print("Input: ")
print(f"X Shape: {X.shape}\n")

# Initialize Feed-Forward Network
feed_forward_network = FeedForwardNetwork(d_model, d_ff)

# Apply Feed-Forward Network
output = feed_forward_network(X)
print("Output: ")
print(f"Output Shape: {output.shape}")
