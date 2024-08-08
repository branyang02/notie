import { cpp } from "@codemirror/lang-cpp";
import { go } from "@codemirror/lang-go";
import { java } from "@codemirror/lang-java";
import { javascript } from "@codemirror/lang-javascript";
import { python } from "@codemirror/lang-python";
import { rust } from "@codemirror/lang-rust";
import { indentUnit } from "@codemirror/language";
import { duotoneLight } from "@uiw/codemirror-theme-duotone";
import { tokyoNightStorm } from "@uiw/codemirror-theme-tokyo-night-storm";
import CodeMirror, { ReactCodeMirrorRef } from "@uiw/react-codemirror";
import {
    Button,
    Card,
    Code,
    IconButton,
    Pane,
    PlayIcon,
    ResetIcon,
    Spinner,
} from "evergreen-ui";
import { useCallback, useRef, useState } from "react";

import { runCode, RunCodeResponse } from "../service/api";
import CodeHeader from "./CodeHeader";

const getLanguageCode = (language: string) => {
    switch (language) {
        case "c":
        case "cpp":
            return cpp();
        case "go":
            return go();
        case "java":
            return java();
        case "rust":
            return rust();
        case "javascript":
        case "js":
        case "typescript":
            return javascript();
        default:
            return python();
    }
};

const CodeBlock = ({
    initialCode,
    language = "python",
    darkMode = false,
}: {
    initialCode: string;
    language?: string;
    darkMode?: boolean;
}) => {
    const [isLoading, setIsLoading] = useState(false);
    const [output, setOutput] = useState("");
    const [error, setError] = useState(false);
    const [code, setCode] = useState(initialCode);
    const [image, setImage] = useState("");
    const editorRef = useRef<ReactCodeMirrorRef>(null);

    const languageCode = getLanguageCode(language);

    const onChange = useCallback((value: string) => {
        setCode(value);
    }, []);

    const runCodeAsync = async () => {
        setIsLoading(true);
        try {
            const data: RunCodeResponse = await runCode(code, language);
            if (
                data.output.trim().startsWith("Traceback") ||
                data.output.trim().startsWith("File") ||
                data.output.trim().startsWith("Exception") ||
                data.output.toLowerCase().includes("error") ||
                data.output.toLowerCase().includes("core dumped")
            ) {
                setError(true);
            } else {
                setError(false);
            }
            setOutput(data.output);
            if (data.image !== "") {
                setImage(data.image);
            }
        } catch (error) {
            setOutput(`Execution failed: ${error}`);
            setError(true);
            setImage("");
        } finally {
            setIsLoading(false);
        }
    };

    const clearOutput = () => {
        setOutput("");
        setImage("");
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

    const CodeEditor = (
        <CodeMirror
            ref={editorRef}
            value={initialCode}
            extensions={[languageCode, indentUnit.of("    ")]}
            maxHeight="800px"
            minHeight="100px"
            theme={darkMode ? tokyoNightStorm : duotoneLight}
            onChange={onChange}
        />
    );

    const ResetButton = (
        <IconButton
            size="small"
            appearance="minimal"
            icon={ResetIcon}
            intent="danger"
            onClick={resetEditor}
        />
    );

    const RunButton = (
        <Button
            iconAfter={PlayIcon}
            appearance="primary"
            intent="success"
            isLoading={isLoading}
            onClick={runCodeAsync}
        >
            Run Code
        </Button>
    );

    const OutputBox = (
        <Pane
            position="relative"
            borderRadius={8}
            overflow="hidden"
            marginBottom={16}
        >
            <Card
                background="gray50"
                padding={16}
                elevation={1}
                borderRadius={8}
                style={{
                    maxHeight: "500px",
                    overflowY: "auto",
                }}
            >
                <Pane>
                    <Button
                        appearance="minimal"
                        intent="danger"
                        onClick={clearOutput}
                        style={{ float: "right" }}
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
                            color={error ? "red" : "black"}
                            style={{
                                wordBreak: "break-word",
                                overflowWrap: "break-word",
                                whiteSpace: "pre-wrap",
                            }}
                        >
                            {output}
                        </Code>
                        {image && (
                            <img
                                src={`data:image/png;base64,${image}`}
                                alt="Output"
                                style={{
                                    maxWidth: "100%",
                                    marginBottom: "10px",
                                }}
                            />
                        )}
                    </>
                )}
            </Card>
        </Pane>
    );

    return (
        <Pane>
            <CodeHeader language={language} code={code} darkMode={darkMode} />
            <Pane>
                <Pane
                    position="relative"
                    overflow="hidden"
                    marginBottom={16}
                    style={{ borderRadius: "0 0 10px 10px" }}
                >
                    {CodeEditor}
                    <Pane position="absolute" top={0} right={0} padding={8}>
                        {ResetButton}
                    </Pane>
                    <Pane position="absolute" bottom={0} right={0} padding={8}>
                        {RunButton}
                    </Pane>
                </Pane>
                {(isLoading || output || image) && OutputBox}
            </Pane>
        </Pane>
    );
};

export default CodeBlock;
