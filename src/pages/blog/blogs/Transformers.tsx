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
import remarkSlug from 'remark-slug';

import CodeBlock from '../../../components/CodeBlock';
import encoder_only_transformer from './transformer-code/encoder_only_transformer.py?raw';
import ffn from './transformer-code/ffn.py?raw';
import multi_head_attention from './transformer-code/multi_head_attention.py?raw';
import positional_encoding from './transformer-code/positional_encoding.py?raw';
import scaled_dot_product_attention from './transformer-code/scaled_dot_product_attention.py?raw';
import self_attention from './transformer-code/self_attention.py?raw';
import transformersMd from './Transformers.md?raw';

type CodeProps = React.HTMLAttributes<HTMLElement> & { children?: React.ReactNode };

const components = {
  code({ children }: CodeProps) {
    return <CodeBlock initialCode={String(children)} />;
  },
};

function processMarkdown(markdownContent: string): string {
  return markdownContent
    .replace(/\$\{encoder_only_transformer\}/g, encoder_only_transformer)
    .replace(/\$\{ffn\}/g, ffn)
    .replace(/\$\{multi_head_attention\}/g, multi_head_attention)
    .replace(/\$\{positional_encoding\}/g, positional_encoding)
    .replace(/\$\{self_attention\}/g, self_attention)
    .replace(/\$\{scaled_dot_product_attention\}/g, scaled_dot_product_attention);
}

const markdownContent = processMarkdown(transformersMd);

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
          // eslint-disable-next-line react/no-children-prop
          remarkPlugins={[remarkGfm, remarkMath]}
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
