import { Button, Card, Code, Pane, PlayIcon } from 'evergreen-ui';
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
      await new Promise((resolve) => setTimeout(resolve, 2000));
      // await fetch('https://api.example.com/run-code', {
      //   method: 'POST',
      //   body: JSON.stringify({ code }),
      // });
      const simulatedResult = 'Execution result: Success!';
      setOutput(simulatedResult);
      setError(false);
    } catch (error) {
      // Handle errors if the API call fails
      setOutput(`Execution failed: ${error}`);
      setError(true);
    } finally {
      // Ensure loading state is cleared after execution or error
      setIsLoading(false);
    }
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
      {output && (
        <Card background="tint1" padding={16} elevation={1} borderRadius={8}>
          <Code appearance="minimal" color={error ? 'red' : 'black'}>
            {output}
          </Code>
        </Card>
      )}
    </Pane>
  );
};

export default CodeBlock;
