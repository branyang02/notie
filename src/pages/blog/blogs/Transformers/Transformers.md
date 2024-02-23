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
8. [Encoder-Only Transformer](#encoder-only-transformer)
9. [Decoder Block](#decoder-block)
10. [Decoder-Only Transformer](#decoder-only-transformer)

</details>

#### **Introduction**

There are **_a lot_** of resources out there that explain transformers[^1], but I wanted to write my own blog post to explain transformers in a way that I understand.

**This blog post is NOT**:

- A high-level comprehensive guide to transformers. For that, I recommend reading [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar.
- A tutorial on how to use Transformers in Hugging Face's [transformers](https://huggingface.co/transformers/) library. For that, I recommend reading the [official documentation](https://huggingface.co/transformers/).
- A tutorial on how to use the _nn.Transformer_ module in PyTorch. For that, I recommend reading the [official documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html). Instead, we will be implementing the transformer model from scratch using basic PyTorch operations.
- A tutorial on how to train a transformer model. We will only cover the arthitecture, and how each component is interconnected.
- Showcase the performance of the transformer model on a specific task.

**This blog post is**:

- **A mathematical explaination** of the transformer model and its components, where I will clearly define each component and explain how they are interconnected.
- Contains **live, runnable** code snippets to show how to implement the transformer model in **PyTorch** using basic operations.

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

Note that in [Figure 2](#fig2), we have a optional mask that can be applied to the attention weights. This is used in the decoder layers of the transformer to prevent the model from looking at future tokens in the sequence, therefore creating autoregressive generation of the output sequence. We implment this by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections. [Figure 3](#fig3) shows the masked attention mechanism.

![](https://branyang02.github.io/images/masked-attention.png)
<span id="fig3"
class="caption">Fig. 3: Masked Scaled Dot-Product Attention. (Source: Yu Meng, <a href="https://yumeng5.github.io/teaching/2024-spring-cs6501">UVA CS 6501 NLP</a>)
</span>

The following code snippet shows how to implement a simple scaled dot-product attention in PyTorch. We will use smaller sequence length and dimension sizes for demonstration purposes.

```execute
${scaled_dot_product_attention}
```

##### **Multi-Head Attention**

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

```execute
${multi_head_attention}
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

#### **Encoder-Only Transformer**

The following code snippet shows how to implement the encoder block in PyTorch by using the components we have defined earlier:

```execute
${encoder_only_transformer}
```

#### **Decoder Block**

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

#### **Decoder-Only Transformer**

To perform auto-regressive generation using a decoder-only transformer, we need to convert the last decoder block output $\textbf{Y}_{\text{N\_dec}}'$ to a probability distribution over the vocabulary. We can do this by applying a linear transformation followed by a softmax activation function to the output $\textbf{Y}_{\text{N\_dec}}'$.

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

```execute
${decoder_only_transformer}
```

Note that this example is purely for inference and architectural demonstration purposes only, and weights are randomized. In practice, the weights are learned during training.

#### **Conclusion**

Now you are familiar with the transformer architecture and its components! We have covered the input embeddings, positional encoding, attention mechanism, multi-head attention mechanism, add & norm layer, feed-forward network, encoder block, and decoder block. We have also implemented the encoder-only transformer and decoder-only transformer in PyTorch using basic operations.

I hope this blog post has helped you understand the transformer architecture better. If you have any questions or feedback, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/byang02/), or my email at [jqm9ba@virginia.edu](mailto: jqm9ba@virginia.edu).

[^1]: Vaswani, A., et al. "Attention is all you need," in Advances in neural information processing systems, vol. 30, 2017.
