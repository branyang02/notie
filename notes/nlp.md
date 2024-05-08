# Blog: Transformers: A Mathematical Explanation and Implementation in PyTorch

<span class="subtitle">
Date: 2/20/2024 | Author: Brandon Yang
</span>

## **Introduction**

There are **_a lot_** of resources out there that explain transformers[^1], but I wanted to write my own blog post to explain transformers in a way that I understand.

All provided codes in this blog post are available in the [GitHub repository](https://github.com/branyang02/personal_website/tree/main/src/pages/blog/blogs/Transformers/transformers-code).

**This blog post is NOT**:

- A high-level comprehensive guide to transformers. For that, I recommend reading [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar.
- A tutorial on how to use Transformers in Hugging Face's [transformers](https://huggingface.co/transformers/) library. For that, I recommend reading the [official documentation](https://huggingface.co/transformers/).
- A tutorial on how to use the _nn.Transformer_ module in PyTorch. For that, I recommend reading the [official documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html). Instead, we will be implementing the transformer model from scratch using basic PyTorch operations.
- A tutorial on how to train a transformer model. We will only cover the architecture, and how components are interconnected.
- To showcase the performance of the transformer model. For that, I recommend reading the [original paper](https://arxiv.org/abs/1706.03762) by Vaswani et al.

**This blog post is**:

- **A mathematical explanation** of the transformer model and its components, where I will clearly define each component and explain how they are interconnected.
- Contains **live, runnable** code snippets to show how to implement the transformer model in **PyTorch** using basic operations.

## **Architecture Overview**

<img src="https://branyang02.github.io/images/transformer.png" width="50%" height="auto" margin="20px auto" display="block">
<span id="fig1"
class="caption">Fig. 1: The transformer architecture
</span>

## **Input Embeddings**

The transformer takes in a sequence of tokens, and converts each token into a vector representation.
Token embeddings are learned during training, and are used to represent the input tokens.
Check out Hugging Face's [tokenizer tutorial](https://huggingface.co/transformers/tokenizer_summary.html) for more information on tokenization.

In this blog post, we define the token embeddings as a matrix $$\textbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$$, where
$$n$$ is the number of words (sequence length) in the input sequence and $$d_{\text{model}}$$ is the dimension of the input embeddings.

## **Positional Encoding**

Next, to capture the position of each token in the sequence, we add positional encodings to the token embeddings.
The positional encoding matrix $$\textbf{PE} \in \mathbb{R}^{n \times d_{\text{model}}}$$ is defined as:

$$
\begin{align*}
\textbf{PE}(pos, 2i) &= \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right) \\
\textbf{PE}(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\end{align*}
$$

where $$\textbf{PE}(pos, 2i)$$ and $$\textbf{PE}(pos, 2i+1)$$ are the $$2i$$-th and $$(2i+1)$$-th dimensions of the positional encoding at position $$pos$$, respectively.

Therefore, we can think of positional encoding as a function that maps the position of each token to a unique vector in the $$d_{\text{model}}$$-dimensional space.

$$
\textbf{PE} =
\begin{bmatrix}
\textbf{PE}(1, 1) & \textbf{PE}(1, 2) & \cdots & \textbf{PE}(1, d_{\text{model}}) \\
\textbf{PE}(2, 1) & \textbf{PE}(2, 2) & \cdots & \textbf{PE}(2, d_{\text{model}}) \\
\vdots & \vdots & \ddots & \vdots \\
\textbf{PE}(n, 1) & \textbf{PE}(n, 2) & \cdots & \textbf{PE}(n, d_{\text{model}})
\end{bmatrix}
$$

Finally, we add the positional encoding to the token embeddings to get the input embeddings:

$$
\textbf{Z} = \textbf{X} + \textbf{PE} \quad \text{where} \quad \textbf{Z} \in \mathbb{R}^{n \times d_{\text{model}}}
$$

The following code snippet shows how to implement a simple positional encoding in PyTorch:

```execute-python
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
```

## **Attention**

The attention mechanism is a key component of the transformer architecture.
It can be described as mapping a **query** and a set of **key-value** pairs to an output, where the query, keys, values, and output are all vectors.

First, we compute $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ matrices from the input embeddings $\mathbf{Z}$:

$$
\begin{align*}
\mathbf{Q} &= \mathbf{Z} \mathbf{W}^Q \\
\mathbf{K} &= \mathbf{Z} \mathbf{W}^K \\
\mathbf{V} &= \mathbf{Z} \mathbf{W}^V
\end{align*}
$$

where $\mathbf{W}^Q, \mathbf{W}^K, \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $\mathbf{W}^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ are the query, key, and value weight matrices, respectively,
and they are learned during training. $$d_k$$ and $$d_v$$ are the dimensions of the query and value vectors, respectively.

Therefore, the sizes of the matrices are:

$$
\begin{align*}
\mathbf{Q} &\in \mathbb{R}^{n \times d_k} \\
\mathbf{K} &\in \mathbb{R}^{n \times d_k} \\
\mathbf{V} &\in \mathbb{R}^{n \times d_v}
\end{align*}
$$

Let's see what this looks like in code:

```execute-python
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
```

First we look at the **_scaled dot-product attention_** mechanism, and then we will look at the **_multi-head attention_** mechanism.

### **Scaled Dot-Product Attention**

<img src="https://branyang02.github.io/images/scaled-dot-product.png" width="30%" height="auto" margin="20px auto" display="block">
<span id="fig2"
class="caption">Fig. 2: Scaled Dot-Product Attention
</span>

The scaled dot-product attention mechanism is defined as a function $$\text{Attention} : \mathbb{R}^{n \times d_k} \times \mathbb{R}^{n \times d_k} \times \mathbb{R}^{n \times d_v} \rightarrow \mathbb{R}^{n \times d_v}$$

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

Note that in [Figure 2](#fig2), we have a optional mask that can be applied to the attention weights. This is used in the decoder layers of the transformer to prevent the model from looking at future tokens in the sequence, therefore creating autoregressive generation of the output sequence. We implment this by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections. [Figure 3](#fig3) shows the masked attention mechanism.

![](https://branyang02.github.io/images/masked-attention.png)
<span id="fig3"
class="caption">Fig. 3: Masked Scaled Dot-Product Attention. (Source: Yu Meng, <a href="https://yumeng5.github.io/teaching/2024-spring-cs6501">UVA CS 6501 NLP</a>)
</span>

The following code snippet shows how to implement a simple scaled dot-product attention in PyTorch. We will use smaller sequence length and dimension sizes for demonstration purposes.

```execute-python
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
```

### **Multi-Head Attention**

The multi-head attention mechanism is a way to allow the model to focus on different parts of the input sequence at the same time.
It consists of $h$ parallel attention layers, where each layer is called a **head**.
Each head has its own query, key, and value weight matrices, which are learned during training.
The output of each head is concatenated and linearly transformed to produce the final output.

<img src="https://branyang02.github.io/images/MHA.png" width="30%" height="auto" margin="20px auto" display="block">
<span id="fig4"
class="caption">Fig. 4: Multi-Head Attention
</span>

Therefore, we need to redefine the query, key, and value weight matrices for each head:

$$
\begin{align*}
\mathbf{W}^Q_i &\in \mathbb{R}^{d_{\text{model}} \times d_k} \\
\mathbf{W}^K_i &\in \mathbb{R}^{d_{\text{model}} \times d_k} \\
\mathbf{W}^V_i &\in \mathbb{R}^{d_{\text{model}} \times d_v}
\end{align*}
$$

where $i = 1, 2, \ldots, h$. In many Transformer implementations, $d_k = d_v = d_{\text{model}} / h$, where $h$ is the number of heads.

This means that the query, key, and value matrices for each head are computed as:

$$
\begin{align*}
\mathbf{Q}_i &= \mathbf{Z} \mathbf{W}^Q_i \\
\mathbf{K}_i &= \mathbf{Z} \mathbf{W}^K_i \\
\mathbf{V}_i &= \mathbf{Z} \mathbf{W}^V_i
\end{align*}
$$

The output of each head is computed as:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) \quad \text{for} \quad i = 1, 2, \ldots, h
$$

The output of the multi-head attention mechanism is a function $$\text{MultiHeadAttention} : \mathbb{R}^{n \times d_{\text{model}}} \rightarrow \mathbb{R}^{n \times d_{\text{model}}}$$

$$
\text{MultiHeadAttention}(\mathbf{Z}) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)\mathbf{W}^O
$$

where $\mathbf{W}^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ is a learned weight matrix.

Therefore, putting the Attention mechanism and the Multi-Head Attention mechanism together, we get the following code snippet:

```execute-python
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
```

## **Add & Norm**

The Add & Norm layer is a **residual connection** followed by **layer normalization**.

### **Residual Connection**

A residual connection involves adding the input of a sub-layer
(e.g., self-attention or feed-forward network) to its output before passing it to the next layer.
This technique helps in mitigating the vanishing gradient problem in deep networks, allowing gradients
to flow directly through the architecture. The operation can be mathematically represented as follows:

$$
\text{Output} = \text{Input} + \text{SubLayer}(\text{Input})
$$

where $$\text{SubLayer}(\cdot)$$ is an operation performed by either the self-attention mechanism or the feed-forward network within the transformer block.

### **Layer Normalization**

After adding the input and output of the sub-layer,
layer normalization is applied. Layer normalization involves computing the mean and variance used for normalization
across the features (not across the batch as in batch normalization) for each data point individually.
It stabilizes the learning process and improves the training speed and effectiveness.

Layer normalization is a function $$\text{LayerNorm} : \mathbb{R}^{n \times d_{\text{model}}} \rightarrow \mathbb{R}^{n \times d_{\text{model}}}$$

$$
\text{LayerNorm}(\mathbf{X}) = \frac{\mathbf{X} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

where $$\mu$$ and $$\sigma$$ are the mean and standard deviation of the input tensor $$\mathbf{X}$$, respectively.
$\epsilon$ is a small constant added for numerical stability,
and $\gamma$ and $\beta$ are learnable parameters of the layer normalization that allow for rescaling and recentering the normalized values.

Therefore, to perform normalization with residual connection, we can simply add the residual connection to the output of the sub-layer and apply layer normalization to the sum.

$$
\text{LayerNorm}(\text{Input} + \text{SubLayer}(\text{Input}))
$$

## **Feed-Forward Network**

The feed-forward network consists of two linear transformations with a ReLU activation function in between.
It can be definied as a function $$\text{FFN} : \mathbb{R}^{n \times d_{\text{model}}} \rightarrow \mathbb{R}^{n \times d_{\text{model}}}$$

$$
\text{FFN}(\mathbf{X}) = \text{ReLU}(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

where $$\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}, \mathbf{b}_1 \in \mathbb{R}^{d_{\text{ff}}}, \mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}, \mathbf{b}_2 \in \mathbb{R}^{d_{\text{model}}},$$
and $$d_{\text{ff}}$$ is the dimension of the feed-forward network.

The following code snippet shows how to implement a simple feed-forward network in PyTorch:

```execute-python
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
```

## **Encoder Block**

We have now fully covered all components needed to build the Encoder block of the transformer.

<img src="https://branyang02.github.io/images/encoder-only.jpg" width="30%" height="auto" margin="20px auto" display="block">
<span id="fig5"
class="caption">Fig. 5: Encoder Block
</span>

The encoder block consists of the following components:

1. **Input Embeddings**: We start with an input $\textbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$, where $n$ is the number of words (sequence length) in the input sequence and $d_{\text{model}}$ is the dimension of the input embeddings.
   $$
   \textbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}
   $$
2. **Positional Encoding**: We add positional encodings to the token embeddings to capture the position of each token in the sequence to get the input embeddings $\textbf{Z}$:
   $$
   \textbf{Z} = \textbf{X} + \textbf{PE}, \quad \textbf{Z} \in \mathbb{R}^{n \times d_{\text{model}}}, \quad \textbf{PE} \in \mathbb{R}^{n \times d_{\text{model}}}
   $$
3. **Multi-Head Attention**: We apply the multi-head attention mechanism to the input embeddings $\textbf{Z}$ to get the attention output $\textbf{A}$:
   $$
   \textbf{A} = \text{MultiHeadAttention}(\textbf{Z}), \quad \textbf{A} \in \mathbb{R}^{n \times d_{\text{model}}}
   $$
4. **Add & Norm**: We add the input embeddings $\textbf{Z}$ to the attention output $\textbf{A}$ and apply layer normalization to get the normalized output $\textbf{N}$:
   $$
   \textbf{N} = \text{LayerNorm}(\textbf{Z} + \textbf{A}), \quad \textbf{N} \in \mathbb{R}^{n \times d_{\text{model}}}
   $$
5. **Feed-Forward Network**: We apply the feed-forward network to the normalized output $\textbf{N}$ to get the feed-forward output $\textbf{O}$:
   $$
   \textbf{O} = \text{FFN}(\textbf{N}), \quad \textbf{O} \in \mathbb{R}^{n \times d_{\text{model}}}
   $$
6. **Add & Norm**: We add the normalized output $\textbf{N}$ to the feed-forward output $\textbf{O}$ and apply layer normalization to get the final output $\textbf{Z}'$:
   $$
   \textbf{Z}' = \text{LayerNorm}(\textbf{N} + \textbf{O}), \quad \textbf{Z}' \in \mathbb{R}^{n \times d_{\text{model}}}
   $$

After we get the output $\textbf{Z}'$ from a single encoder block, we reuse it as the input to the next encoder block without passing through the positional encoding again. Suppose we have $\text{N\_enc}$ encoder blocks, we can denote the output of the $k$-th encoder block as $\textbf{Z}_k'$, where $k = 1, 2, \ldots, \text{N\_enc}$.

## **Encoder-Only Transformer**

The following code snippet shows how to implement the encoder block in PyTorch by using the components we have defined earlier:

```execute-python
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
```

## **Decoder Block**

The decoder block is similar to the encoder block, but with an additional multi-head attention mechanism that takes the output of the encoder block as input. The decoder is also autoregressive, meaning that it generates the output sequence one token at a time, and uses the previously generated tokens as input to generate the next token.

The decoder block consists of the following components:

1.  **Input Embeddings**: Suppose we have already generated the first $i$ tokens of the output sequence. We start with an input $\textbf{Y} \in \mathbb{R}^{i \times d_{\text{model}}}$, where $i$ is the number of tokens generated so far, and $d_{\text{model}}$ is the dimension of the input embeddings.
    $$
    \textbf{Y} \in \mathbb{R}^{i \times d_{\text{model}}}
    $$
2.  **Positional Encoding**: We add positional encodings to the token embeddings to capture the position of each token in the sequence to get the input embeddings $\textbf{Z}$:

    $$
    \textbf{Z} = \textbf{Y} + \textbf{PE}, \quad \textbf{Z} \in \mathbb{R}^{i \times d_{\text{model}}}, \quad \textbf{PE} \in \mathbb{R}^{i \times d_{\text{model}}}
    $$

3.  **Masked Multi-Head Attention**: We apply the masked multi-head attention mechanism to the input embeddings $\textbf{Z}$ to get the attention output $\textbf{A}$. This means in addition to passing the $\textbf{Q}, \textbf{K}, \textbf{V}$ matrices to the scaled dot-product attention, we also pass a mask to the attention mechanism, where

    $$
    \text{mask}_{i \times i} = \begin{matrix}
    \begin{bmatrix}
    0 & -\infty & \cdots & -\infty \\
    0 & 0 & \cdots & -\infty \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & 0
    \end{bmatrix}
    \end{matrix}
    $$

    Therefore, for a single head, the attention weights are computed as:

    $$
    \text{Attention}(\textbf{Q}, \textbf{K}, \textbf{V}) = \text{softmax}\left(\frac{\textbf{Q}\textbf{K}^T}{\sqrt{d_k}} + \text{mask}\right)\textbf{V}
    $$

    where $\textbf{Q}, \textbf{K} \in \mathbb{R}^{i \times d_k}$, $\textbf{V} \in \mathbb{R}^{i \times d_v}$, and $\text{mask} \in \mathbb{R}^{i \times i}$.

    Finally, the output of the masked multi-head attention mechanism is computed as:

    $$
    \textbf{A} = \text{MultiHeadAttention}(\textbf{Z}, \text{mask}), \quad \textbf{A} \in \mathbb{R}^{i \times d_{\text{model}}}
    $$

4.  **Add & Norm**: We add the input embeddings $\textbf{Z}$ to the attention output $\textbf{A}$ and apply layer normalization to get the normalized output $\textbf{N}$:

    $$
    \textbf{N} = \text{LayerNorm}(\textbf{Z} + \textbf{A}), \quad \textbf{N} \in \mathbb{R}^{i \times d_{\text{model}}}
    $$

5.  **Multi-Head Attention**: We apply the multi-head attention mechanism to the normalized output $\textbf{N}$:

    $$
    \textbf{A}' = \text{MultiHeadAttention}(\textbf{N}), \quad \textbf{A}' \in \mathbb{R}^{i \times d_{\text{model}}}
    $$

    Optionally, if we are constructing a encoder-decoder architecture, we can also pass the output of the final encoder block to the multi-head attention mechanism to get the attention output $\textbf{A}'$:

    $$
    \textbf{A}' = \text{MultiHeadAttention}(\textbf{N}, \textbf{Z}'_{N\_enc}), \quad \textbf{A}' \in \mathbb{R}^{i \times d_{\text{model}}}
    $$

    where $\textbf{Z}'_{N\_enc}$ is the output of the final encoder block. In this case, the scaled dot-product attention takes the **query** from the previous layer $\textbf{N}$ and the **key** and **value** from the output of the encoder block $\textbf{Z}'_{N\_enc}$.

6.  **Add & Norm**: We add the normalized output $\textbf{N}$ to the attention output $\textbf{A}'$ and apply layer normalization to get the output $\textbf{M}$:

    $$
    \textbf{M} = \text{LayerNorm}(\textbf{N} + \textbf{A}'), \quad \textbf{M} \in \mathbb{R}^{i \times d_{\text{model}}}
    $$

7.  **Feed-Forward Network**: We apply the feed-forward network to the output $\textbf{M}$ to get the feed-forward output $\textbf{O}$:

    $$
    \textbf{O} = \text{FFN}(\textbf{M}), \quad \textbf{O} \in \mathbb{R}^{i \times d_{\text{model}}}
    $$

8.  **Add & Norm**: We add the output $\textbf{M}$ to the feed-forward output $\textbf{O}$ and apply layer normalization to get the final output $\textbf{Y}'$:
    $$
    \textbf{Y}' = \text{LayerNorm}(\textbf{M} + \textbf{O}), \quad \textbf{Y}' \in \mathbb{R}^{i \times d_{\text{model}}}
    $$

After we get output $\textbf{Y}'$ from a single decoder block, we reuse it as the input to the next decoder block without passing through the positional encoding again. Suppose we have $\text{N\_dec}$ decoder blocks, we can denote the output of the $k$-th decoder block as $\textbf{Y}_k'$, where $k = 1, 2, \ldots, \text{N\_dec}$.

## **Decoder-Only Transformer**

To perform autoregressive generation using a decoder-only transformer, we need to convert the last decoder block output $\textbf{Y}_{\text{N\_dec}}'$ to a probability distribution over the vocabulary. We can do this by applying a linear transformation followed by a softmax activation function to the output $\textbf{Y}_{\text{N\_dec}}'$.

![](https://i.stack.imgur.com/bWnx0.png)
<span id="fig6"
class="caption">Fig. 6: Linear transformation followed by a softmax activation function. (Source: <a href="https://ai.stackexchange.com/questions/40179/how-does-the-decoder-only-transformer-architecture-work">Stack Exchange</a>)

This can be done in the following steps:

1. Given final decoder block output $\textbf{Y}_{\text{N\_dec}}' \in \mathbb{R}^{i \times d_{\text{model}}}$, where $i$ is the number of tokens generated so far, and $d_{\text{model}}$ is the dimension of the input embeddings, we apply a linear transformation to get the logits:
   $$
   \textbf{L} = \textbf{Y}_{\text{N\_dec}}'\textbf{W}^L + \textbf{b}^L
   $$
   where $\textbf{W}^L \in \mathbb{R}^{d_{\text{model}} \times V}$ and $\textbf{b}^L \in \mathbb{R}^{V}$ are the weight and bias matrices of the linear transformation, respectively, and $V$ is the size of the vocabulary.
2. We apply a softmax activation function to the logits to get the probability distribution over the vocabulary:
   $$
   \textbf{P} = \text{softmax}(\textbf{L})
   $$
   where $\textbf{P} \in \mathbb{R}^{i \times V}$ is the probability distribution over the vocabulary.
3. We can then sample the next token from the probability distribution $\textbf{P}$ to generate the next token in the output sequence.

This process is repeated until the end-of-sequence token is generated, or until a maximum sequence length is reached.

The following code snippet shows how to implement the decoder block in PyTorch by using the components we have defined earlier:

```execute-python
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
```

Note that this example is purely for inference and architectural demonstration purposes only, and weights are randomized. In practice, the weights are learned during training.

## **Conclusion**

Now, you should be familiar with the transformer architecture and its components! We have covered the input embeddings, positional encoding, attention mechanism, multi-head attention mechanism, add & norm layer, feed-forward network, encoder block, and decoder block. We have also implemented the encoder-only transformer and decoder-only transformer in PyTorch using basic operations.

I hope this blog post has helped you understand the transformer architecture better. If you have any questions or feedback, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/byang02/), or my email at [jqm9ba@virginia.edu].

[^1]: Vaswani, A., et al. "Attention is all you need," in Advances in neural information processing systems, vol. 30, 2017.
