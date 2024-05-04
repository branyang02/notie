import '../styles/note-toc.css';

import { Pane } from 'evergreen-ui';
import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeSlug from 'rehype-slug';

const generateTableOfContents = (markdownContent: string) => {
  let res = '# Contents\n---\n';
  const pattern = /^#+ (.*)$/gm;
  const matches = markdownContent.match(pattern);
  for (const match of matches || []) {
    const level = match.match(/^#+/)?.[0].length || 0;
    if (level === 1) continue;
    const title = match.replace(/^#+|\*+/g, '').trim();
    const id = title
      .replace(/\s+/g, '-')
      .toLowerCase()
      .replace(/[+.()']/g, '');
    res += `${'\t'.repeat(level - 2)}-  [${title}](#${id})\n`;
  }
  return res;
};

const NoteToc = ({ markdownContent }: { markdownContent: string }) => {
  const toc = useMemo(() => generateTableOfContents(markdownContent), [markdownContent]);

  return (
    <Pane padding="20px" className="note-toc">
      <ReactMarkdown rehypePlugins={[[rehypeSlug, { prefix: 'toc-' }]]}>
        {toc}
      </ReactMarkdown>
    </Pane>
  );
};

export default NoteToc;
