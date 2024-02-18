import 'katex/dist/katex.min.css'; // Ensure KaTeX CSS is imported to style the equations
import '../styles/blogPost.css';

import { useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

const todaysDate = new Date().toLocaleDateString('en-US', {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
});

const markdownContent = `
# **Vision Transformer (ViT)**
<span class="subtitle">
Date: ${todaysDate} | Author: Brandon Yang
</span>

**Transformers**[^1] have been widely used in natural language processing (NLP) tasks, such as language modeling, translation, 
and summarization. However, they have not been as popular in computer vision tasks. 
Convolutional neural networks (CNNs) have been the dominant architecture for image classification tasks. 
The **Vision Transformer (ViT)**[^2] is a transformer-based model 
that has shown promising results in image classification tasks. This blog post will provide an overview of 
the ViT architecture and its performance on image classification tasks.

## **ViT Architecture**
![](https://1.bp.blogspot.com/-_mnVfmzvJWc/X8gMzhZ7SkI/AAAAAAAAG24/8gW2AHEoqUQrBwOqjhYB37A7OOjNyKuNgCLcBGAsYHQ/s1600/image1.gif)
<span id="fig1" 
class="caption">Fig. 1: Vision Transformer treats an input image as a sequence of patches. (Source: <a href="https://blog.research.google/2020/12/transformers-for-image-recognition-at.html">Gooble AI Blog</a>)
</span>
#### **Image to Patch Embeddings**
ViT receives an input image instead of a 1D sequence of text token embeddings in a standard transformer architecture.
$$
\\begin{align*}
\\textbf{x} \\in \\mathbb{R}^{H \\times W \\times C}
\\end{align*}
$$
where $H$ is the height, $W$ is the width, and $C$ is the number of channels of the input image.
We flatten the image into a sequence of 2D patches:
$$
\\begin{align*}
\\textbf{x}_p &\\in \\mathbb{R}^{ (P^2 \\times C) \\times N}
\\end{align*}
$$
where $$N = \\frac{HW}{P^2}$$ is the number of patches, and $P \\times P$ is the size of each patch. 

To process these patches using a Transformer, we need to convert the patches into a sequence with 
positional embeddings. This is done by projecting the flattened patch vectors into a 
$$D$$-dimensional embedding space. The transformation is performed using
a learnable linear projection $\\mathbf{E}$. The projection transforms each patch vector
from $\\mathbb{R}^{P^2 \\times C}$ to $\\mathbb{R}^D$.
$$
\\begin{align*}
\\textbf{E} \\in \\mathbb{R}^{(P^2 \\times C) \\times D}
\\end{align*}
$$
Therefore, for each patch $$i \\in [1, N]$$, its flattened vector $$\\textbf{x}_p^i$$ is
transformed via the projection $$\\textbf{E}$$ to produce its embedding $$\\textbf{z}_0^i$$ as follows:
$$
\\begin{align*}
\\textbf{z}_0^i =  \\textbf{x}_p^i \\textbf{E}
\\end{align*}
$$
where $$\\textbf{z}_0^i \\in \\mathbb{R}^D$$. Next, we consider all the patches as a sequence of embeddings
$$\\textbf{Z}_0$$:
$$
\\begin{align*}
\\textbf{Z}_0 = \\left[\\mathbf{x}_{class},   \\textbf{z}_0^1, \\textbf{z}_0^2, \\ldots, \\textbf{z}_0^N\\right] + \\textbf{E}_{pos}
\\end{align*}
$$
Here, $$\\textbf{Z}_0$$ is the initial input matrix to the transformer encoder, containing:

* The class token $$\\mathbf{x}_{class}$$, which is a learnable parameter representing the entire image.
* The sequence of $$N$$ patch embeddings that have been projected into the $$D$$-dimensional space.
* The positional embeddings $$\\textbf{E}_{pos} \\in \\mathbb{R}^{(N + 1) \\times D}$$, 
which are added to each patch embedding and the class token embedding.


#### **Transformer Encoder**
Next, we feed our text embeddings to a Transformer encoder, which consists of Multi-Head 
Self-Attention (MSA) and Feedforward Neural Network (FNN) as shown in [_Fig. 2_](#fig2).
![Example Image](https://branyang02.github.io/images/transformer-encoder.jpg "Placeholder Image")
<span id="fig2" 
class="caption">Fig. 1: Standard Transformer Encoder. (Source: Yu Meng, <a href="https://yumeng5.github.io/teaching/2024-spring-cs6501">UVA CS 6501 NLP</a>)
</span>
First we pass through the Multi-Head Self-Attention (MSA) layer, which computes the attention
weights for each token in the sequence. The MSA layer is defined as follows:
$$
\\begin{align*}
\\textbf{Z}^\\prime_l = \\text{MSA}\\left(\\text{LN}(\\textbf{Z}_{l-1})\\right) + \\textbf{Z}_{l-1} \\quad \\text{for} \\quad l = 1, 2, \\ldots, L
\\end{align*}
$$
where $$\\textbf{Z}_{l-1}$$ is the input to the $$l$$-th layer, $$\\textbf{Z}^\\prime_l$$ is the output of the $$l$$-th layer,
$$\\text{LN}$$ is the layer normalization, and $$L$$ is the number of layers in the transformer encoder.

The MSA layer is followed by a Feedforward Neural Network (FNN) layer, which applies a pointwise feedforward transformation
to each token in the sequence independently. The FNN layer is defined as follows:
$$
\\begin{align*}
\\textbf{Z}_l = \\text{FNN}\\left(\\text{LN}(\\textbf{Z}^\\prime_l)\\right) + \\textbf{Z}^\\prime_l \\quad \\text{for} \\quad l = 1, 2, \\ldots, L
\\end{align*}
$$
where $$\\textbf{Z}^\\prime_l$$ is the input to the $$l$$-th layer, $$\\textbf{Z}_l$$ is the output of the $$l$$-th layer.

Finally, the output of the last layer of the transformer encoder is used for classification.
$$
\\begin{align*}
\\textbf{y} = \\text{softmax}\\left(\\text{LN}(\\textbf{Z}_L)\\right)
\\end{align*}
$$
where $$\\textbf{y}$$ represents the class probabilities given the input image in a multi-class classification task.





[^1]: Vaswani, A., et al. "Attention is all you need," in Advances in neural information processing systems, vol. 30, 2017.
[^2]: Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale," in arXiv preprint arXiv:2010.11929, 2020.

`;

const Blog = () => {
  useEffect(() => {
    const footnotesTitle = document.querySelector('.footnotes h2');
    if (footnotesTitle) {
      footnotesTitle.innerHTML = '<strong>References</strong>';
    }
  }, []);

  return (
    <div className="blog-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeRaw]}
        // eslint-disable-next-line react/no-children-prop
        children={markdownContent}
      />
    </div>
  );
};

export default Blog;
