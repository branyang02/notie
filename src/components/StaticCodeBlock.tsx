import { CodeBlock, nord } from 'react-code-blocks';

const StaticCodeBlock = ({ code, language }: { code: string; language: string }) => {
  return (
    <CodeBlock text={code} language={language} showLineNumbers={false} theme={nord} />
  );
};

export default StaticCodeBlock;
