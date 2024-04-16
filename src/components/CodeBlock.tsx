import { cpp } from '@codemirror/lang-cpp';
import { python } from '@codemirror/lang-python';
import { indentUnit } from '@codemirror/language';
import { tokyoNightStorm } from '@uiw/codemirror-theme-tokyo-night-storm';
import CodeMirror, { ReactCodeMirrorRef } from '@uiw/react-codemirror';
import {
  Button,
  Card,
  Code,
  IconButton,
  Pane,
  PlayIcon,
  ResetIcon,
  Spinner,
} from 'evergreen-ui';
import { useCallback, useRef, useState } from 'react';

import { runCCode, RunCodeResponse, runPythonCode } from '../service/api';

interface CodeBlockProps {
  initialCode: string;
  language?: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ initialCode, language = 'python' }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [output, setOutput] = useState('');
  const [error, setError] = useState(false);
  const [code, setCode] = useState(initialCode);
  const [image, setImage] = useState('');
  const editorRef = useRef<ReactCodeMirrorRef>(null);

  const runCode = language === 'c' ? runCCode : runPythonCode;
  const languageCode = language === 'c' ? cpp() : python();

  const onChange = useCallback((value: string) => {
    setCode(value);
  }, []);

  const runCodeAsync = async () => {
    setIsLoading(true);
    try {
      const data: RunCodeResponse = await runCode(code);
      if (
        data.output.trim().startsWith('Traceback') ||
        data.output.trim().startsWith('File') ||
        data.output.trim().startsWith('Exception') ||
        data.output.toLowerCase().includes('error')
      ) {
        setError(true);
      } else {
        setError(false);
      }
      setOutput(data.output);
      if (data.image !== '') {
        setImage(data.image);
      }
    } catch (error) {
      setOutput(`Execution failed: ${error}`);
      setError(true);
      setImage('');
    } finally {
      setIsLoading(false);
    }
  };

  const clearOutput = () => {
    setOutput('');
    setImage('');
    setError(false);
    setIsLoading(false);
  };

  const resetEditor = () => {
    setCode(initialCode);
    if (editorRef.current?.view) {
      const { state } = editorRef.current.view;
      const end = state.doc.length;
      editorRef.current.view.dispatch({
        changes: { from: 0, to: end, insert: initialCode },
      });
    }
  };

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
      <Pane>
        <Pane
          position="relative"
          overflow="hidden"
          marginBottom={16}
          style={{ borderRadius: '0 0 10px 10px' }}
        >
          <CodeMirror
            ref={editorRef}
            value={initialCode}
            extensions={[languageCode, indentUnit.of('    ')]}
            height="500px"
            theme={tokyoNightStorm}
            onChange={onChange}
          />
          <Pane position="absolute" top={0} right={0} padding={8}>
            <IconButton
              size="small"
              appearance="minimal"
              icon={ResetIcon}
              intent="danger"
              onClick={resetEditor}
            />
          </Pane>
          <Pane position="absolute" bottom={0} right={0} padding={8}>
            <Button
              iconAfter={PlayIcon}
              appearance="primary"
              intent="success"
              isLoading={isLoading}
              onClick={runCodeAsync}
            >
              Run Code
            </Button>
          </Pane>
        </Pane>
        {/* Output box */}
        {(isLoading || output || image) && (
          <Pane position="relative" borderRadius={8} overflow="hidden" marginBottom={16}>
            <Card
              background="tint1"
              padding={16}
              elevation={1}
              borderRadius={8}
              style={{
                maxHeight: '500px',
                overflowY: 'auto',
              }}
            >
              <Pane>
                <Button
                  appearance="minimal"
                  intent="danger"
                  onClick={clearOutput}
                  style={{ float: 'right' }}
                >
                  Clear Output
                </Button>
              </Pane>
              {isLoading ? (
                <Spinner />
              ) : (
                <>
                  <Code
                    appearance="minimal"
                    color={error ? 'red' : 'black'}
                    style={{
                      wordBreak: 'break-word',
                      overflowWrap: 'break-word',
                      whiteSpace: 'pre-wrap',
                    }}
                  >
                    {output}
                  </Code>
                  {image && (
                    <img
                      src={`data:image/png;base64,${image}`}
                      alt="Output"
                      style={{ maxWidth: '100%', marginBottom: '10px' }}
                    />
                  )}
                </>
              )}
            </Card>
          </Pane>
        )}
      </Pane>
    </Pane>
  );
};

export default CodeBlock;
