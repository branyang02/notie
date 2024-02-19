import { Button, Card, Pane, Paragraph, PlayIcon, Spinner } from 'evergreen-ui';
import React, { useState } from 'react';
import { CopyBlock, nord } from 'react-code-blocks';

const CodeBlock = ({ code, language }: { code: string; language: string }) => {
  const [isLoading, setIsLoading] = useState(false);
  // State to store the output or error message
  const [output, setOutput] = useState('');
  const [error, setError] = useState(false);

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
        <CopyBlock
          customStyle={{
            height: '500px',
            overflow: 'scroll',
          }}
          text={code}
          language={language}
          showLineNumbers
          theme={nord}
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
          <Card background="tint1" padding={16} elevation={1} borderRadius={8}>
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
                  wordBreak: 'break-word', // Breaks words to prevent overflow
                  overflowWrap: 'break-word', // Allows long words to be able to break and wrap onto the next line
                  whiteSpace: 'pre-wrap', // Maintains whitespace formatting but allows text to wrap
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
