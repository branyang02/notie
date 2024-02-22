# **Transformer: Explained in Math and PyTorch**

<span class="subtitle">
Date: 2/20/2024 | Author: Brandon Yang
</span>

<details><summary>Table of Content</summary>

1. [Architecture Overview](#architecture-overview)
2. [Input Embeddings](#input-embeddings)
3. [Positional Encoding](#positional-encoding)
4. [Attention](#attention)
5. [Add & Norm](#add--norm)
6. [Feed-Forward Network](#feed-forward-network)
7. [Encoder Block](#encoder-block)

</details>

#### **Introduction**

There are **_a lot_** of resources out there that explain transformers, but I wanted
to write my own blog post to help me understand the transformer model better.
In this blog post, I will explain the transformer model and its components, and I will provide
**live**, **runnable** code snippets to show how to implement the transformer model in PyTorch.

#### **Architecture Overview**

<img src="https://branyang02.github.io/images/transformer.png" width="50%" height="auto" margin="20px auto" display="block">
<span id="fig1"
class="caption">Fig. 1: The transformer architecture
</span>

#### **Input Embeddings**

The transformer takes in a sequence of tokens, and converts each token into a vector representation.
Token embeddings are learned during training, and are used to represent the input tokens.
Check out Hugging Face's [tokenizer tutorial](https://huggingface.co/transformers/tokenizer_summary.html) for more information on tokenization.

In this blog post, we define the token embeddings as a matrix $$\textbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$$, where
$$n$$ is the number of words (sequence length) in the input sequence and $$d_{\text{model}}$$ is the dimension of the input embeddings.

#### **Positional Encoding**

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

```execute
${positional_encoding}
```

#### **Attention**

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

```execute
${self_attention}
```

First we look at the **_scaled dot-product attention_** mechanism, and then we will look at the **_multi-head attention_** mechanism.

##### **Scaled Dot-Product Attention**

<img src="https://branyang02.github.io/images/scaled-dot-product.png" width="30%" height="auto" margin="20px auto" display="block">
<span id="fig2"
class="caption">Fig. 2: Scaled Dot-Product Attention
</span>

The scaled dot-product attention mechanism is defined as a function $$\text{Attention} : \mathbb{R}^{n \times d_k} \times \mathbb{R}^{n \times d_k} \times \mathbb{R}^{n \times d_v} \rightarrow \mathbb{R}^{n \times d_v}$$

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

The following code snippet shows how to implement a simple scaled dot-product attention in PyTorch:

```execute
${scaled_dot_product_attention}
```

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

```execute
${multi_head_attention}
```

##### **Masked Multi-Head Attention**

In the decoder block of the transformer, we use a masked multi-head attention mechanism to prevent the model from looking at future tokens in the sequence.
This is done by applying a mask to the scaled dot-product attention mechanism, which sets the attention weights to 0 for the future tokens.

The following code snippet shows how to implement a simple masked multi-head attention in PyTorch:

```execute
${masked_multi_head_attention}
```


#### **Add & Norm**

The Add & Norm layer is a **residual connection** followed by **layer normalization**.

##### **Residual Connection**

A residual connection involves adding the input of a sub-layer
(e.g., self-attention or feed-forward network) to its output before passing it to the next layer.
This technique helps in mitigating the vanishing gradient problem in deep networks, allowing gradients
to flow directly through the architecture. The operation can be mathematically represented as follows:

$$
\text{Output} = \text{Input} + \text{SubLayer}(\text{Input})
$$

where $$\text{SubLayer}(\cdot)$$ is an operation performed by either the self-attention mechanism or the feed-forward network within the transformer block.

##### **Layer Normalization**

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

#### **Feed-Forward Network**

The feed-forward network consists of two linear transformations with a ReLU activation function in between.
It can be definied as a function $$\text{FFN} : \mathbb{R}^{n \times d_{\text{model}}} \rightarrow \mathbb{R}^{n \times d_{\text{model}}}$$

$$
\text{FFN}(\mathbf{X}) = \text{ReLU}(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

where $$\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}, \mathbf{b}_1 \in \mathbb{R}^{d_{\text{ff}}}, \mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}, \mathbf{b}_2 \in \mathbb{R}^{d_{\text{model}}},$$
and $$d_{\text{ff}}$$ is the dimension of the feed-forward network.

The following code snippet shows how to implement a simple feed-forward network in PyTorch:

```execute
${ffn}
```

#### **Encoder Block**

We have now fully covered all components needed to build the Encoder block of the transformer.

<img src="https://branyang02.github.io/images/encoder-only.jpg" width="30%" height="auto" margin="20px auto" display="block">
<span id="fig1"
class="caption">Fig. 1: Encoder Block
</span>

The encoder block consists of the following components:

1. **Input Embeddings**: We start with an input $\textbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$, where $n$ is the number of words (sequence length) in the input sequence and $d_{\text{model}}$ is the dimension of the input embeddings.
2. **Positional Encoding**: We add positional encodings to the token embeddings to capture the position of each token in the sequence to get the input embeddings $\textbf{Z} = \textbf{X} + \textbf{PE}$, where $\textbf{PE} \in \mathbb{R}^{n \times d_{\text{model}}}$ is the positional encoding matrix, and $\textbf{Z} \in \mathbb{R}^{n \times d_{\text{model}}}$.
3. **Multi-Head Attention**: We apply the multi-head attention mechanism to the input embeddings $\textbf{Z}$ to get the attention output $\textbf{A} = \text{MultiHeadAttention}(\textbf{Z})$, where $\textbf{A} \in \mathbb{R}^{n \times d_{\text{model}}}$.
4. **Add & Norm**: We add the input embeddings $\textbf{Z}$ to the attention output $\textbf{A}$ and apply layer normalization to get the normalized output $\textbf{N} = \text{LayerNorm}(\textbf{Z} + \textbf{A})$, where $\textbf{N} \in \mathbb{R}^{n \times d_{\text{model}}}$.
5. **Feed-Forward Network**: We apply the feed-forward network to the normalized output $\textbf{N}$ to get the feed-forward output $\textbf{O} = \text{FFN}(\textbf{N})$, where $\textbf{O} \in \mathbb{R}^{n \times d_{\text{model}}}$.
6. **Add & Norm**: We add the normalized output $\textbf{N}$ to the feed-forward output $\textbf{O}$ and apply layer normalization to get the final output $\textbf{Z}' = \text{LayerNorm}(\textbf{N} + \textbf{O})$, where $\textbf{Z}' \in \mathbb{R}^{n \times d_{\text{model}}}$.

In a encoder-only transformer, the output of the encoder block $\textbf{Z}'$ is then fed to the next encoder block, without passing through the positional encoding again.

The following code snippet shows how to implement the encoder block in PyTorch by using the components we have defined earlier:

```execute
${encoder_only_transformer}
```
