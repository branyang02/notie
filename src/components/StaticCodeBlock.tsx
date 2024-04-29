import { Pane } from 'evergreen-ui';
import { atomOneLight, CopyBlock, nord } from 'react-code-blocks';

import { useDarkMode } from '../context/DarkModeContext';

const StaticCodeBlock = ({ code, language }: { code: string; language: string }) => {
  const { darkMode } = useDarkMode();

  return (
    <Pane>
      <Pane
        className="language-box"
        paddingY={1}
        paddingX={8}
        style={{
          fontSize: '0.8rem',
          borderRadius: '10px 10px 0 0',
          backgroundColor: '#afb8c133',
        }}
      >
        {''}
        {language}
      </Pane>
      <CopyBlock
        text={code}
        language={language}
        showLineNumbers={false}
        theme={darkMode ? nord : atomOneLight}
        startingLineNumber={2}
        customStyle={{ borderRadius: '0 0 10px 10px' }}
      />
    </Pane>
  );
};

export default StaticCodeBlock;
