import { CopyBlock, nord } from 'react-code-blocks';

const StaticCodeBlock = ({ code, language }: { code: string; language: string }) => {
  return (
    <CopyBlock text={code} language={language} showLineNumbers={false} theme={nord} />
  );
};

export default StaticCodeBlock;
