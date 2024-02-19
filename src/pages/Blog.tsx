import 'katex/dist/katex.min.css'; // Ensure KaTeX CSS is imported to style the equations
import '../styles/blogPost.css';

// import 'highlight.js/styles/github.css';
import { useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

import CodeBlock from '../components/CodeBlock';

type CodeProps = React.HTMLAttributes<HTMLElement> & { children?: React.ReactNode };

const todaysDate = new Date().toLocaleDateString('en-US', {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
});

const components = {
  code({ children }: CodeProps) {
    return <CodeBlock code={String(children)} language="python" />;
  },
};

const markdownContent = `
# **A Deep Dive into OpenAI's Sora**
<span class="subtitle">
Date: ${todaysDate} | Author: Brandon Yang
</span>

# **Introduction**
OpenAI's **Sora**[^1] represents a groundbreaking step in video generation, 
leveraging advanced diffusion models and transformer architectures to 
interpret and generate complex visual narratives from textual prompts, images, and videos.
In this blog post, we delve into the _potential_ underlying mechanisms of Sora,
exploring its training, architecture, and capabilities to understand its
remarkable video generation capabilities. As a **closed-source** model, we do not have access to the
exact details of Sora's architecture and training. Therefore, the following
discussion is based on our understanding of the underlying principles of diffusion models and transformers.
![](https://cdn.openai.com/tmp/s/prompting_4.png)

Sora is a **_diffusion transformer_** model, which combines the strengths of diffusion models and transformers to
generate high-quality videos from textual prompts. Before we dive into the details of Sora, let's first
understand the underlying principles of diffusion models and transformers, more specifically, the **Vision Transformer (ViT)**[^3].

## **Vision Transformer (ViT)**
**Transformers**[^2] have been widely used in natural language processing (NLP) tasks, such as language modeling, translation, 
and summarization. However, they have not been as popular in computer vision tasks. 
Convolutional neural networks (CNNs) have been the dominant architecture for image classification tasks. 
The **Vision Transformer (ViT)**[^3] is a transformer-based **_classification_** model 
that has shown promising results. This section will provide an overview of 
the ViT architecture and its performance on image classification tasks.

### **ViT Architecture**
![](https://1.bp.blogspot.com/-_mnVfmzvJWc/X8gMzhZ7SkI/AAAAAAAAG24/8gW2AHEoqUQrBwOqjhYB37A7OOjNyKuNgCLcBGAsYHQ/s1600/image1.gif)
<span id="fig1" 
class="caption">Fig. 1: Vision Transformer treats an input image as a sequence of patches. (Source: <a href="https://blog.research.google/2020/12/transformers-for-image-recognition-at.html">Gooble AI Blog</a>)
</span>
#### **Image Input to Patch Embeddings**
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


#### **Feeding Into Transformer Encoder**
Next, we feed our text embeddings to a Transformer encoder, which consists of Multi-Head 
Self-Attention (MSA) and Feedforward Neural Network (FNN) as shown in [_Fig. 2_](#fig2).
![](https://branyang02.github.io/images/transformer-encoder.jpg "Standard Transformer Encoder")
<span id="fig2" 
class="caption">Fig. 2: Standard Transformer Encoder. (Source: Yu Meng, <a href="https://yumeng5.github.io/teaching/2024-spring-cs6501">UVA CS 6501 NLP</a>)
</span>
First we pass through the Multi-Head Self-Attention (MSA) layer, which computes the attention
weights for each token in the sequence. The MSA layer is defined as follows:
$$
\\begin{align*}
\\textbf{Z}^\\prime_l = \\text{MSA}\\left(\\text{LN}(\\textbf{Z}_{l-1})\\right) + \\textbf{Z}_{l-1} \\quad \\text{for} \\quad l = 1, 2, \\ldots, L
\\end{align*}
$$
<details><summary>Multihead Self-Attention (MSA)</summary>

For each element in an input sequence $\\textbf{z} \\in \\mathbb{R}^{N \\times D}$,
we compute a weighted sum over all values $\\textbf{v}$ in the sequence.
The attention weights $A_{ij}$ are based on the pairwise similarity between 
two elements of the sequence and their respective query $\\textbf{q}^i$ and key $\\textbf{k}^j$ vectors.

$$
\\begin{align*}
[\\textbf{q}, \\textbf{k}, \\textbf{v}] &= \\textbf{z} \\textbf{U}_{q,k,v} \\quad &&\\text{where} \\quad \\textbf{U}_{q,k,v} \\in \\mathbb{R}^{D \\times 3d_{head}} \\\\
A &= \\text{softmax}\\left(\\frac{\\textbf{q} \\textbf{k}^T}{\\sqrt{d_{head}}}\\right) &&\\text{where} \\quad A \\in \\mathbb{R}^{N \\times N} \\\\
\\text{SA}(\\textbf{z}) &= A \\textbf{v} \\quad &&\\text{where} \\quad \\textbf{v} \\in \\mathbb{R}^{N \\times D}
\\end{align*}
$$
where $d_{head}$ is the dimension of the query, key, and value vectors in each head, and $N$ is the number of tokens in the sequence.
MSA is an extention of SA, where we run $k$ self-attention operations, called "heads", in parallel, 
and project their concatenated outputs. The MSA layer is defined as follows:
$$
\\begin{align*}
\\text{MSA}(\\textbf{z}) &= \\left[ \\text{SA}_1(\\textbf{z}), \\text{SA}_2(\\textbf{z}), \\ldots, \\text{SA}_k(\\textbf{z}) \\right] \\textbf{U}_{msa}
\\end{align*}
$$
where $\\textbf{U}_{msa} \\in \\mathbb{R}^{k \\cdot d_{head} \\times D}$ is a learnable parameter.
More details about MSA can be found in the original transformer paper[^2], and the [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar.

</details>

<details><summary>Layer Normalization (LN)</summary>

Layer normalization is used to normalize the input to each layer of the transformer.
It is defined as follows:
$$
\\begin{align*}
\\text{LN}(\\textbf{x}) = \\frac{\\textbf{x} - \\mu}{\\sigma} \\odot \\gamma + \\beta
\\end{align*}
$$
where $$\\mu$$ and $$\\sigma$$ are the mean and standard deviation of the input $$\\textbf{x}$$,
$$\\gamma$$ and $$\\beta$$ are learnable parameters, and $$\\odot$$ denotes element-wise multiplication.
More details and performance about layer normalization can be found in _Ba et al._[^4].
</details>

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
y = \\text{softmax}\\left(\\text{LN}(\\textbf{Z}_L)\\right)
\\end{align*}
$$
where $$y \\in \\textbf{Y}$$ represents the predicted class given the input image and class labels $$\\textbf{Y}$$ in a multi-class classification task.

#### **Training Setup**
Now that we have defined the architecture of the Vision Transformer, we can train the model using
a standard cross-entropy loss function. The loss function is defined as follows:
$$
\\begin{align*}
\\mathcal{L}(\\theta) = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=1}^{C} y_{ij} \\log \\hat{y}_{ij}
\\end{align*}
$$
where $$\\theta$$ are the learnable parameters of the model, $$N$$ is the number of training samples, $$C$$ is the number of classes,
$$y_{ij}$$ is the true label of the $$i$$-th sample for class $$j$$, and $$\\hat{y}_{ij}$$ is the predicted probability of the $$i$$-th sample for class $$j$$.
We can train the model using backpropagation and the Adam optimizer to minimize the loss function.
Here is a list of all parameters that need to be learned during training:
- **Patch Projection Parameters**: $$\\textbf{E} \\in \\mathbb{R}^{(P^2 \\times C) \\times D}$$.
- **Positional Embeddings**: $$\\textbf{E}_{pos} \\in \\mathbb{R}^{(N + 1) \\times D}$$.
- **Class Token**: $$\\textbf{x}_{class} \\in \\mathbb{R}^D$$.
- **Transformer Parameters**: 
  - **Attention Parameters**: For each attention head $i$ in the MSA, $$\\textbf{Q}_i, \\textbf{K}_i, \\textbf{V}_i \\in \\mathbb{R}^{D \\times d_{head}}$$.
  - **Output Projection of MSA**: $$\\textbf{U}_{msa} \\in \\mathbb{R}^{k \\cdot d_{head} \\times D}$$.
- **Feedforward Network Parameters**: Each FNN layer consists of two linear transformations with a ReLU activation in between.
The weight matrices and bias vectors for the first and second linear transformations in the 
$l$-th layer FNN are represented as $$\\textbf{W}_{1,l}, \\textbf{b}_{1,l}, \\textbf{W}_{2,l}, \\textbf{b}_{2,l}$$.
  - $$\\textbf{W}_{1,l} \\in \\mathbb{R}^{D \\times D_{fnn}}, \\textbf{b}_{1,l} \\in \\mathbb{R}^{D_{fnn}}$$.
  - $$\\textbf{W}_{2,l} \\in \\mathbb{R}^{D_{fnn} \\times D}, \\textbf{b}_{2,l} \\in \\mathbb{R}^{D}$$.
  - $$D_{fnn}$$ is the dimension of the hidden layer in the FNN.
- **Layer Normalization Parameters**: For each layer normalization step, the scale ($\\gamma$) and shift ($\\beta$) parameters are learned, where:
  - $$\\gamma, \\beta \\in \\mathbb{R}^D$$.
- **Output Projection Parameters**: The output projection of the last layer of the transformer is represented as $$\\textbf{W}_{out}, \\textbf{b}_{out}$$.
  - $$\\textbf{W}_{out} \\in \\mathbb{R}^{D \\times C}, \\textbf{b}_{out} \\in \\mathbb{R}^C$$.

#### **ViT vs. CNN**
The Vision Transformer has shown promising results in image classification tasks,
outperforming convolutional neural networks (CNNs) on several benchmarks. 
The key advantages of ViT over CNNs are:
- **Scalability**: ViT can handle images of arbitrary size, while CNNs require resizing the input images.
- **Global Context**: ViT captures global context information by treating the input image as a sequence of patches,
while CNNs use local receptive fields to capture spatial information.
- **Fewer Parameters**: ViT has fewer parameters than CNNs, making it more efficient for training and inference.
- **Transfer Learning**: ViT can be fine-tuned on downstream tasks with fewer labeled examples,
while CNNs require a large amount of labeled data to achieve good performance.

![](https://branyang02.github.io/images/vit_performance.png "ViT Performance vs SOTA CNNs")
<span id="fig3"
class="caption">Fig. 3: ViT Performance on ImageNet Classification. (Source: Dosovitskiy et al.[^3])
</span>

## **Diffusion Models**
Now that we have discussed the Vision Transformer, a model that can _classify_ images, we can move on to diffusion models, models that can _generate_ images.
Diffusion models are a class of **_generative models_** that learn to generate high-quality images from a sequence of noise vectors.
The key idea behind diffusion models is to _diffuse_ the noise vectors to generate realistic images.
We will be discussing the difussion process and the U-Net architecture introduced in the paper 
_Denoising Diffusion Probabilistic Models_ by Ho et al.[^5].

#### **Problem Definition**
Let's start by defining the problem of image generation.
Suppose we have a dataset of images $$\\textbf{X} = \\{\\textbf{x}_1, \\textbf{x}_2, \\ldots, \\textbf{x}_N\\}$$, where $$\\textbf{x}_i \\in \\mathbb{R}^{H \\times W \\times C}$$.
Suppose we also know the underlying distribution $$q(\\textbf{x})$$ of the training data.
Our goal is to fit a probabilistic model $$p(\\textbf{x})$$ to the data such that it can generate realistic images $$\\textbf{x}$$ that are similar to the training data.

To learn the distribution $$p(\\textbf{x})$$, we follow a two-step process:
1. **Forward Process**: We continuously add Gaussian noise to the input image to _destroy_ the image.
2. **Backward Process**: We continuously _denoise_ the noisy image to _recover_ the original image.

![](https://www.assemblyai.com/blog/content/images/2022/05/image.png)
![](https://www.assemblyai.com/blog/content/images/2022/05/image-1.png)
<span id="fig4"
class="caption">Fig. 4: Foward(Top) and Backward(Bottom) Process. (Source:  <a href="https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/">AssemblyAI Blog</a>)
</span>

The first step 
#### **Forward Process**
The forward process, aka the _noise process_, is when we add Gaussian noise to the input image to _destroy_ the image.
We sample a training data point at random $$\\textbf{x}_0 \\sim q(\\textbf{x})$$  and progressively add more noise 
to the data point to geenerate a sequence of noisy images $$\\textbf{x}_t$$, where $$t = 1, 2, \\ldots, T$$.


\`\`\`execute
test = []
for i in range(10):
    test.append(i)
print(test)
\`\`\`


[^1]: Brooks, Peebles, et al., "Video generation models as world simulators,", 2024.
[^2]: Vaswani, A., et al. "Attention is all you need," in Advances in neural information processing systems, vol. 30, 2017.
[^3]: Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale," in arXiv preprint arXiv:2010.11929, 2020.
[^4]: J. Ba, J. Kiros, G. Hinton. "Layer normalization," in arXiv preprint arXiv:1607.06450, 2016.
[^5]: J. Ho, A. Jain, P. Abbeel. "Denoising diffusion probabilistic models," in Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.
[^6]: R. O'Connor, "Introduction to Diffusion Models for Machine Learning," in AssemblyAI Blog, 2022.
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
        rehypePlugins={[rehypeKatex, rehypeRaw, rehypeHighlight]}
        // eslint-disable-next-line react/no-children-prop
        components={components}
        // eslint-disable-next-line react/no-children-prop
        children={markdownContent}
      />
    </div>
  );
};

export default Blog;
