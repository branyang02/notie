import 'katex/dist/katex.min.css';
import '../../../styles/blogPost.css';

import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

import CodeBlock from '../../../components/CodeBlock';
import StaticCodeBlock from '../../../components/StaticCodeBlock';
import TikZ from '../../../components/TikZ';
import markdown from './cso2.md?raw';

type CodeProps = React.HTMLAttributes<HTMLElement> & {
  node?: unknown;
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
};

const components = {
  code({ inline, className, children, ...props }: CodeProps) {
    const match = /\w+/.exec(className || '');

    if (!inline && match) {
      const language = className?.split('language-').pop() || '';
      const content = Array.isArray(children) ? children.join('') : children;
      const code = String(content).replace(/\n$/, '');
      if (language.includes('execute-')) {
        return <CodeBlock initialCode={code} language={language.split('-').pop()} />;
      }
      if (language === 'tikz') {
        return <TikZ tikzScript={code} />;
      }
      return <StaticCodeBlock code={code} language={language} />;
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
  const pattern = /```(\w+)/g;
  const processedContent = markdownContent.replace(pattern, '```language-$1');

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
