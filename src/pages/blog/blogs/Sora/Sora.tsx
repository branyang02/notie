import 'katex/dist/katex.min.css'; // Ensure KaTeX CSS is imported to style the equations
import '../../../../styles/blogPost.css';

import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

import CodeBlock from '../../../../components/CodeBlock';
import soraMD from './Sora.md?raw';
import forward_diffusion from './sora-code/forward_diffusion.py?raw';

type CodeProps = React.HTMLAttributes<HTMLElement> & { children?: React.ReactNode };

const components = {
  code({ children }: CodeProps) {
    return <CodeBlock initialCode={String(children)} />;
  },
};

function processMarkdown(markdownContent: string): string {
  return markdownContent.replace(/\$\{forward_diffusion\}/g, forward_diffusion);
}

const markdownContent = processMarkdown(soraMD);

const Sora = () => {
  return (
    <div className="blog-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeRaw, rehypeHighlight, rehypeSlug]}
        components={components}
      >
        {markdownContent}
      </ReactMarkdown>
    </div>
  );
};

export default Sora;
