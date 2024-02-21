import 'katex/dist/katex.min.css'; // Ensure KaTeX CSS is imported to style the equations
import '../../../styles/blogPost.css';

import { IconButton, LightbulbIcon, MoonIcon } from 'evergreen-ui';
import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkToc from 'remark-toc';

import CodeBlock from '../../../components/CodeBlock';

type CodeProps = React.HTMLAttributes<HTMLElement> & { children?: React.ReactNode };

const todaysDate = new Date().toLocaleDateString('en-US', {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
});

const components = {
  code({ children }: CodeProps) {
    return <CodeBlock initialCode={String(children)} />;
  },
};

const markdownContent = `
# **Transformers**
<span class="subtitle">
Date: ${todaysDate} | Author: Brandon Yang
</span>

## Contents

## **Introduction**
There are **_a lot_** of resources out there that explain transformers, but I wanted
to create a blog post that explains transformers in a way that I understand. This blog post will include
mathematical derivation of each component of the transformer, and will also include PyTorch code examples.

#### **Architecture Overview**
<img src="https://branyang02.github.io/images/transformer.png" width="50%" height="auto" margin="20px auto" display="block">
<span id="fig1"
class="caption">Fig. 1: The transformer architecture
</span>

#### **Input Embeddings**
The transformer takes in a sequence of tokens, and converts each token into a vector representation.
Token embeddings are learned during training, and are used to represent the input tokens.
Check out Hugging Face's [Tokenizer Tutorial](https://huggingface.co/transformers/tokenizer_summary.html) for more information on tokenization.

In this blog post, we define the token embeddings as a matrix $$\\textbf{X} \\in \\mathbb{R}^{n \\times d_{\\text{model}}}$$, where
$$n$$ is the number of words (sequence length) in the input sequence and $$d_{\\text{model}}$$ is the dimension of the input embeddings.

#### **Positional Encoding**
Next, to capture the position of each token in the sequence, we add positional encodings to the token embeddings.
The positional encoding matrix $$\\textbf{PE} \\in \\mathbb{R}^{n \\times d_{\\text{model}}}$$ is defined as:
$$
\\begin{align*}
\\textbf{PE}(pos, 2i) &= \\sin\\left(\\frac{pos}{10000^{\\frac{2i}{d_{\\text{model}}}}}\\right) \\\\
\\textbf{PE}(pos, 2i+1) &= \\cos\\left(\\frac{pos}{10000^{\\frac{2i}{d_{\\text{model}}}}}\\right)
\\end{align*}
$$
where $$\\textbf{PE}(pos, 2i)$$ and $$\\textbf{PE}(pos, 2i+1)$$ are the $$2i$$-th and $$(2i+1)$$-th dimensions of the positional encoding at position $$pos$$, respectively.

Therefore, we can think of positional encoding as a function that maps the position of each token to a unique vector in the $$d_{\\text{model}}$$-dimensional space.
$$
\\textbf{PE} =
\\begin{bmatrix}
\\textbf{PE}(1, 1) & \\textbf{PE}(1, 2) & \\cdots & \\textbf{PE}(1, d_{\\text{model}}) \\\\
\\textbf{PE}(2, 1) & \\textbf{PE}(2, 2) & \\cdots & \\textbf{PE}(2, d_{\\text{model}}) \\\\
\\vdots & \\vdots & \\ddots & \\vdots \\\\
\\textbf{PE}(n, 1) & \\textbf{PE}(n, 2) & \\cdots & \\textbf{PE}(n, d_{\\text{model}})
\\end{bmatrix}
$$

Finally, we add the positional encoding to the token embeddings to get the input embeddings:
$$
\\textbf{Z} = \\textbf{X} + \\textbf{PE} \\quad \\text{where} \\quad \\textbf{Z} \\in \\mathbb{R}^{n \\times d_{\\text{model}}}
$$

The following code snippet shows how to implement a simple positional encoding in PyTorch:

\`\`\`execute
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
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
print(f"Token Embeddings Shape: {token_embeddings.shape}\\n")

# Initialize Positional Encoding
pos_encoder = PositionalEncoding(d_model)

# Apply Positional Encoding to Token Embeddings
token_embeddings_with_pos = pos_encoder(token_embeddings)
print("Output: ")
print(f"Token Embeddings with Positional Encoding Shape: {token_embeddings_with_pos.shape}")
\`\`\`

#### **Attention**
The attention mechanism is a key component of the transformer architecture.
It can be described as mapping a **query** and a set of **key-value** pairs to an output, where the query, keys, values, and output are all vectors.

First, we compute $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}$ matrices from the input embeddings $\\mathbf{Z}$:
$$
\\begin{align*}
\\mathbf{Q} &= \\mathbf{Z} \\mathbf{W}^Q \\\\
\\mathbf{K} &= \\mathbf{Z} \\mathbf{W}^K \\\\
\\mathbf{V} &= \\mathbf{Z} \\mathbf{W}^V
\\end{align*}
$$
where $\\mathbf{W}^Q, \\mathbf{W}^K, \\in \\mathbb{R}^{d_{\\text{model}} \\times d_k}$ and $\\mathbf{W}^V \\in \\mathbb{R}^{d_{\\text{model}} \\times d_v}$ are the query, key, and value weight matrices, respectively,
and they are learned during training. $$d_k$$ and $$d_v$$ are the dimensions of the query and value vectors, respectively.

Therefore, the sizes of the matrices are:
$$
\\begin{align*}
\\mathbf{Q} &\\in \\mathbb{R}^{n \\times d_k} \\\\
\\mathbf{K} &\\in \\mathbb{R}^{n \\times d_k} \\\\
\\mathbf{V} &\\in \\mathbb{R}^{n \\times d_v}
\\end{align*}
$$

Let's see what this looks like in code:

\`\`\`execute
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
print(f"Z Shape: {Z.shape}\\n")

# Initialize Self-Attention
self_attention = SelfAttention(d_model, d_k, d_v)

# Apply Self-Attention
Q, K, V = self_attention(Z)
print("Output: ")
print(f"Q Shape: {Q.shape} \\nK Shape: {K.shape} \\nV Shape: {V.shape}")
\`\`\`

First we look at the **_scaled dot-product attention_** mechanism, and then we will look at the **_multi-head attention_** mechanism.

##### **Scaled Dot-Product Attention**
<img src="https://branyang02.github.io/images/scaled-dot-product.png" width="30%" height="auto" margin="20px auto" display="block">
<span id="fig2"
class="caption">Fig. 2: Scaled Dot-Product Attention
</span>

The scaled dot-product attention mechanism is defined as a function $$\\text{Attention} : \\mathbb{R}^{n \\times d_k} \\times \\mathbb{R}^{n \\times d_k} \\times \\mathbb{R}^{n \\times d_v} \\rightarrow \\mathbb{R}^{n \\times d_v}$$
$$
\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{softmax}\\left(\\frac{\\mathbf{Q}\\mathbf{K}^T}{\\sqrt{d_k}}\\right)\\mathbf{V}
$$

The following code snippet shows how to implement a simple scaled dot-product attention in PyTorch:

\`\`\`execute
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
d_k = d_v = 64 # Dimension of the key and value
seq_length = 32  # Length of the input sequence

# Initialize Q, K, V
Q = torch.randn(seq_length, d_k)
K = torch.randn(seq_length, d_k)
V = torch.randn(seq_length, d_v)
print("Input: ")
print(f"Q Shape: {Q.shape} \\nK Shape: {K.shape} \\nV Shape: {V.shape}\\n")

# Initialize Scaled Dot-Product Attention
scaled_dot_product_attention = ScaledDotProductAttention()

# Apply Scaled Dot-Product Attention
attention_output = scaled_dot_product_attention(Q, K, V)
print("Output: ")
print(f"Scaled Dot-Product Attention Output Shape: {attention_output.shape}")
\`\`\`

##### **Multi-Head Attention**
The multi-head attention mechanism is a way to allow the model to focus on different parts of the input sequence at the same time.
It consists of $h$ parallel attention layers, where each layer is called a **head**.
Each head has its own query, key, and value weight matrices, which are learned during training.
The output of each head is concatenated and linearly transformed to produce the final output.

<img src="https://branyang02.github.io/images/MHA.png" width="30%" height="auto" margin="20px auto" display="block">
<span id="fig3"
class="caption">Fig. 3: Multi-Head Attention
</span>

Therefore, we need to redefine the query, key, and value weight matrices for each head:
$$
\\begin{align*}
\\mathbf{W}^Q_i &\\in \\mathbb{R}^{d_{\\text{model}} \\times d_k} \\\\
\\mathbf{W}^K_i &\\in \\mathbb{R}^{d_{\\text{model}} \\times d_k} \\\\
\\mathbf{W}^V_i &\\in \\mathbb{R}^{d_{\\text{model}} \\times d_v}
\\end{align*}
$$
where $i = 1, 2, \\ldots, h$. In many Transformer implementations, $d_k = d_v = d_{\\text{model}} / h$, where $h$ is the number of heads.

This means that the query, key, and value matrices for each head are computed as:
$$
\\begin{align*}
\\mathbf{Q}_i &= \\mathbf{Z} \\mathbf{W}^Q_i \\\\
\\mathbf{K}_i &= \\mathbf{Z} \\mathbf{W}^K_i \\\\
\\mathbf{V}_i &= \\mathbf{Z} \\mathbf{W}^V_i
\\end{align*}
$$

The output of each head is computed as:
$$
\\text{head}_i = \\text{Attention}(\\mathbf{Q}_i, \\mathbf{K}_i, \\mathbf{V}_i) \\quad \\text{for} \\quad i = 1, 2, \\ldots, h
$$

The output of the multi-head attention mechanism is a function $$\\text{MultiHeadAttention} : \\mathbb{R}^{n \\times d_{\\text{model}}} \\rightarrow \\mathbb{R}^{n \\times d_{\\text{model}}}$$
$$
\\text{MultiHeadAttention}(\\mathbf{Z}) = \\text{Concat}(\\text{head}_1, \\text{head}_2, \\ldots, \\text{head}_h)\\mathbf{W}^O
$$
where $\\mathbf{W}^O \\in \\mathbb{R}^{hd_v \\times d_{\\text{model}}}$ is a learned weight matrix.

Therefore, putting the Attention mechanism and the Multi-Head Attention mechanism together, we get the following code snippet:

\`\`\`execute
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
        self.attentions = [SelfAttention(d_model, d_k, d_v) for _ in range(n_heads)]
        self.scaled_dot_product_attentions = [
            ScaledDotProductAttention() for _ in range(n_heads)
        ]

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
print(f"Z Shape: {Z.shape}\\n")

# Initialize Multi-Head Attention
multi_head_attention = MultiHeadAttention(d_model, d_k, d_v, h)

# Forward Pass
output = multi_head_attention(Z)
print("Output: ")
print(f"Output Shape: {output.shape}")
\`\`\`

#### **Add & Norm**
The Add & Norm layer is a **residual connection** followed by **layer normalization**.

##### **Residual Connection**
A residual connection involves adding the input of a sub-layer
(e.g., self-attention or feed-forward network) to its output before passing it to the next layer.
This technique helps in mitigating the vanishing gradient problem in deep networks, allowing gradients
to flow directly through the architecture. The operation can be mathematically represented as follows:
$$
\\text{Output} = \\text{Input} + \\text{SubLayer}(\\text{Input})
$$
where $$\\text{SubLayer}(\\cdot)$$ is an operation performed by either the self-attention mechanism or the feed-forward network within the transformer block.

##### **Layer Normalization**
After adding the input and output of the sub-layer,
layer normalization is applied. Layer normalization involves computing the mean and variance used for normalization
across the features (not across the batch as in batch normalization) for each data point individually.
It stabilizes the learning process and improves the training speed and effectiveness.

Layer normalization is a function $$\\text{LayerNorm} : \\mathbb{R}^{n \\times d_{\\text{model}}} \\rightarrow \\mathbb{R}^{n \\times d_{\\text{model}}}$$
$$
\\text{LayerNorm}(\\mathbf{X}) = \\frac{\\mathbf{X} - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta
$$
where $$\\mu$$ and $$\\sigma$$ are the mean and standard deviation of the input tensor $$\\mathbf{X}$$, respectively.
$\\epsilon$ is a small constant added for numerical stability,
and $\\gamma$ and $\\beta$ are learnable parameters of the layer normalization that allow for rescaling and recentering the normalized values.

#### **Feed-Forward Network**
The feed-forward network consists of two linear transformations with a ReLU activation function in between.
It can be definied as a function $$\\text{FFN} : \\mathbb{R}^{n \\times d_{\\text{model}}} \\rightarrow \\mathbb{R}^{n \\times d_{\\text{model}}}$$
$$
\\text{FFN}(\\mathbf{X}) = \\text{ReLU}(\\mathbf{X}\\mathbf{W}_1 + \\mathbf{b}_1)\\mathbf{W}_2 + \\mathbf{b}_2
$$
where $$\\mathbf{W}_1 \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{\\text{ff}}}, \\mathbf{b}_1 \\in \\mathbb{R}^{d_{\\text{ff}}}, \\mathbf{W}_2 \\in \\mathbb{R}^{d_{\\text{ff}} \\times d_{\\text{model}}}, \\mathbf{b}_2 \\in \\mathbb{R}^{d_{\\text{model}}},$$
and $$d_{\\text{ff}}$$ is the dimension of the feed-forward network.

`;

const Transformers = () => {
  const [darkMode, setDarkMode] = useState(
    new Date().getHours() >= 18 || new Date().getHours() < 6,
  );

  useEffect(() => {
    const footnotesTitle = document.querySelector('.footnotes h2');
    if (footnotesTitle) {
      footnotesTitle.innerHTML = '<strong>References</strong>';
    }
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

  return (
    <div style={{ position: 'relative' }}>
      <IconButton
        height={56}
        icon={darkMode ? LightbulbIcon : MoonIcon}
        onClick={() => setDarkMode(!darkMode)}
        style={{
          position: 'fixed',
          bottom: '20px',
          left: '20px',
        }}
      />
      <div className="blog-content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath, remarkToc]}
          rehypePlugins={[rehypeKatex, rehypeRaw, rehypeHighlight]}
          // eslint-disable-next-line react/no-children-prop
          components={components}
          // eslint-disable-next-line react/no-children-prop
          children={markdownContent}
        />
      </div>
    </div>
  );
};

export default Transformers;
