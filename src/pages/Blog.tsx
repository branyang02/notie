import 'katex/dist/katex.min.css'; // Ensure KaTeX CSS is imported to style the equations
import '../styles/blogPost.css';

import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

// Updated markdown content with the table that includes images
const markdownContent = `
# **Vision Transformer (ViT)**

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

\`\`\`
\\begin{align}
a &= b + c \\\\
x &= y - z \\\\
\\mathbb{N} &\\subseteq \\mathbb{Z} \\\\
\\end{align}
\`\`\`

## Images

Here's another example image:

![Example Image](https://branyang02.github.io/images/Brandon_big.jpg "Placeholder Image")

Enjoy writing and rendering Markdown with advanced features!
`;

const Blog = () => {
  return (
    <div className="blog-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        children={markdownContent}
      />
    </div>
  );
};

export default Blog;
