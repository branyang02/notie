import { python } from '@codemirror/lang-python';
import { nord } from '@uiw/codemirror-theme-nord';
import CodeMirror from '@uiw/react-codemirror';
import { Button, Card, Pane, Paragraph, PlayIcon, Spinner } from 'evergreen-ui';
import { useCallback, useState } from 'react';

const CodeBlock = ({ initialCode }: { initialCode: string }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [output, setOutput] = useState('');
  const [error, setError] = useState(false);
  const [code, setCode] = useState(initialCode);

  const onChange = useCallback((value: string) => {
    setCode(value);
  }, []);

  const runCode = async () => {
    setIsLoading(true);
    try {
      console.log(code);
      const result = await fetch(
        'https://yang-website-backend-c3338735a47f.herokuapp.com/api/run-code',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ code }),
        },
      );
      const data = await result.json();
      setOutput(data.output);
      setError(false);
    } catch (error) {
      setOutput(`Execution failed: ${error}`);
      setError(true);
    } finally {
      setIsLoading(false);
    }
  };

  const clearOutput = () => {
    setOutput('');
    setError(false);
    setIsLoading(false);
  };

  return (
    <Pane>
      <Pane position="relative" borderRadius={8} overflow="hidden" marginBottom={16}>
        <CodeMirror
          value={code}
          extensions={[python()]}
          height="500px"
          theme={nord}
          onChange={onChange}
        />
        <Pane position="absolute" bottom={0} right={0} padding={8}>
          <Button
            iconAfter={PlayIcon}
            appearance="primary"
            intent="success"
            isLoading={isLoading}
            onClick={runCode}
          >
            Run Code
          </Button>
        </Pane>
      </Pane>
      {/* Output box */}
      {(isLoading || output) && (
        <Pane position="relative" borderRadius={8} overflow="hidden" marginBottom={16}>
          <Card
            background="tint1"
            padding={16}
            elevation={1}
            borderRadius={8}
            style={{
              maxHeight: '300px',
              overflowY: 'auto',
            }}
          >
            <Pane position="absolute" top={0} right={0} padding={8}>
              <Button appearance="minimal" intent="danger" onClick={clearOutput}>
                Clear Output
              </Button>
            </Pane>

            {isLoading ? (
              <Spinner />
            ) : (
              <Paragraph
                color={error ? 'red' : 'black'}
                style={{
                  wordBreak: 'break-word',
                  overflowWrap: 'break-word',
                  whiteSpace: 'pre-wrap',
                }}
              >
                {output}
              </Paragraph>
            )}
          </Card>
        </Pane>
      )}
    </Pane>
  );
};

export default CodeBlock;
