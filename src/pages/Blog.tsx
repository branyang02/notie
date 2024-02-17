import 'katex/dist/katex.min.css'; // Ensure KaTeX CSS is imported to style the equations
import '../styles/blogPost.css';

import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import supersub from 'remark-supersub';

const markdownContent = `
# **Vision Transformer (ViT)**

**Transformers**<sup>[[1]](https://arxiv.org/abs/1706.03762)</sup> 
have been widely used in natural language processing (NLP) tasks, such as language modeling, translation, 
and summarization. However, they have not been as popular in computer vision tasks. 
Convolutional neural networks (CNNs) have been the dominant architecture for image classification tasks. 
The **Vision Transformer (ViT)** <sup>[[2]](https://arxiv.org/abs/2010.11929)</sup> is a transformer-based model 
that has shown promising results in image classification tasks. This blog post will provide an overview of 
the ViT architecture and its performance on image classification tasks.

## **ViT Architecture**
ViT receives an input image instead of a 1D sequence of text token embeddings.
$$
\\begin{align*}
\\textbf{x} \\in \\mathbb{R}^{H \\times W \\times C}
\\end{align*}
$$
where $H$ is the height, $W$ is the width, and $C$ is the number of channels of the input image.
We faltten the image into a sequence of 2D patches:
$$
\\begin{align*}
\\textbf{x} \\in \\mathbb{R}^{N \\times (P \\times P \\times C)}
\\end{align*}
$$
where $N$ is the number of patches, and $P \\times P$ is the size of each patch.


# References


Welcome to the Markdown features demonstration.

## Basic Syntax

- **Bold text** with \`**\` or \`__\`
- *Italic text* with \`*\` or \`_\`
- ~~Strikethrough~~ with \`~~\`

## GitHub Flavored Markdown (GFM)

This includes:

- Tables
- Task lists
- Strikethrough
- Autolinked URLs

### Adding a Link

Here is an example of adding a link to [OpenAI's Website](https://openai.com "OpenAI Homepage").

### Table Example

| Branch  | Commit           | Image                          |
| ------- | ---------------- | ------------------------------ |
| main    | 0123456789abcdef | ![Example Image](https://via.placeholder.com/150 "Placeholder Image") |
| staging | fedcba9876543210 | ![Example Image](https://via.placeholder.com/150 "Placeholder Image") |

### Task List

- [x] Completed task
- [ ] Incomplete task

## Math Equations

Inline math: $E = mc^2$

Block math:

$$
\\begin{align}
a &= b + c \\\\
x &= y - z \\\\
\\mathbb{N} &\\subseteq \\mathbb{Z}
\\end{align}
$$

## Images

Here's another example image:

![Example Image](https://branyang02.github.io/images/Brandon_big.jpg "Placeholder Image")

Enjoy writing and rendering Markdown with advanced features!
`;

const Blog = () => {
  return (
    <div className="blog-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, supersub]}
        rehypePlugins={[rehypeKatex, rehypeRaw]}
        // eslint-disable-next-line react/no-children-prop
        children={markdownContent}
      />
    </div>
  );
};

export default Blog;
