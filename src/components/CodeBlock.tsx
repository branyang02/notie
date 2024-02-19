import { CopyBlock, nord } from 'react-code-blocks';

const CodeBlock = ({ code, language }: { code: string; language: string }) => {
  return (
    <CopyBlock text={code} language={language} showLineNumbers={true} theme={nord} />
  );
};

export default CodeBlock;
