import 'katex/dist/katex.min.css';
import '../../../styles/blogPost.css';

import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

import StaticCodeBlock from '../../../components/StaticCodeBlock';
import markdown from './cso2.md?raw';

type CodeProps = React.HTMLAttributes<HTMLElement> & {
  node?: unknown;
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
};

const components = {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  code({ node, inline, className, children, ...props }: CodeProps) {
    const match = /\w+/.exec(className || '');

    if (!inline && match) {
      const language = className?.split('language-').pop() || '';
      const content = Array.isArray(children) ? children.join('') : children;
      return (
        <StaticCodeBlock code={String(content).replace(/\n$/, '')} language={language} />
      );
    } else {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }
  },
};

function processMarkdown(markdownContent: string): string {
  let processedContent = markdownContent;
  const cPattern = /```c/g;
  processedContent = processedContent.replace(cPattern, '```language-c');

  const bashPattern = /```bash/g;
  processedContent = processedContent.replace(bashPattern, '```language-bash');

  return processedContent;
}

const markdownContent = processMarkdown(markdown);

const CSO2 = () => {
  return (
    <div className="blog-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeRaw, rehypeHighlight, rehypeSlug]}
        components={components}
        // eslint-disable-next-line react/no-children-prop
        children={markdownContent}
      />
    </div>
  );
};

export default CSO2;
