import 'katex/dist/katex.min.css'; // Ensure KaTeX CSS is imported to style the equations
import '../../../../styles/blogPost.css';

import { IconButton, LightbulbIcon, MoonIcon } from 'evergreen-ui';
import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

import CodeBlock from '../../../../components/CodeBlock';
import transformersMd from './Transformers.md?raw';
import decoder_only_transformer from './transformers-code/decoder_only_transformer.py?raw';
import encoder_only_transformer from './transformers-code/encoder_only_transformer.py?raw';
import ffn from './transformers-code/ffn.py?raw';
import multi_head_attention from './transformers-code/multi_head_attention.py?raw';
import positional_encoding from './transformers-code/positional_encoding.py?raw';
import scaled_dot_product_attention from './transformers-code/scaled_dot_product_attention.py?raw';
import self_attention from './transformers-code/self_attention.py?raw';

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
    .replace(/\$\{scaled_dot_product_attention\}/g, scaled_dot_product_attention)
    .replace(/\$\{decoder_only_transformer\}/g, decoder_only_transformer);
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
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeRaw, rehypeHighlight, rehypeSlug]}
          components={components}
          // eslint-disable-next-line react/no-children-prop
          children={markdownContent}
        />
      </div>
    </div>
  );
};

export default Transformers;
