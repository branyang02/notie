import { cpp } from "@codemirror/lang-cpp";
import { go } from "@codemirror/lang-go";
import { java } from "@codemirror/lang-java";
import { javascript } from "@codemirror/lang-javascript";
import { python } from "@codemirror/lang-python";
import { rust } from "@codemirror/lang-rust";
import { indentUnit } from "@codemirror/language";
import * as themes from "@uiw/codemirror-themes-all";
import { Extension } from "@codemirror/state";
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
    theme,
    copyButtonHoverColor,
}: {
    initialCode: string;
    language?: string;
    theme: string;
    copyButtonHoverColor?: string;
}) => {
    let selectedTheme = themes[theme as keyof typeof themes] as Extension;
    if (!selectedTheme) {
        console.error(`Invalid theme name: ${theme}, falling back to default.`);
        selectedTheme = themes.duotoneLight; // Default fallback theme
    }

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

    const runCodeAsync = useCallback(async () => {
        setIsLoading(true);
        try {
            const data: RunCodeResponse = await runCode(code, language);
            setError(
                data.output.toLowerCase().includes("error") ||
                    data.output.toLowerCase().includes("exception"),
            );
            setOutput(data.output);
            setImage(data.image);
        } catch (error) {
            setOutput(`Execution failed: ${error}`);
            setError(true);
            setImage("");
        } finally {
            setIsLoading(false);
        }
    }, [code, language]);

    const clearOutput = useCallback(() => {
        setOutput("");
        setImage("");
        setError(false);
        setIsLoading(false);
    }, []);

    const resetEditor = useCallback(() => {
        setCode(initialCode);
        if (editorRef.current?.view) {
            const { state } = editorRef.current.view;
            const end = state.doc.length;
            editorRef.current.view.dispatch({
                changes: { from: 0, to: end, insert: initialCode },
            });
        }
    }, [initialCode]);

    return (
        <Pane>
            <CodeHeader
                language={language}
                code={code}
                copyButtonHoverColor={copyButtonHoverColor}
            />
            <Pane>
                <Pane
                    position="relative"
                    overflow="hidden"
                    style={{ borderRadius: "0 0 10px 10px" }}
                >
                    <div className="code-blocks">
                        <CodeMirror
                            ref={editorRef}
                            value={initialCode}
                            extensions={[languageCode, indentUnit.of("    ")]}
                            maxHeight="80vh"
                            minHeight="100px"
                            theme={selectedTheme}
                            onChange={onChange}
                        />
                    </div>
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
                {(isLoading || output || image) && (
                    <OutputBox
                        isLoading={isLoading}
                        output={output}
                        error={error}
                        image={image}
                        clearOutput={clearOutput}
                    />
                )}
            </Pane>
        </Pane>
    );
};

const OutputBox = ({
    isLoading,
    output,
    error,
    image,
    clearOutput,
}: {
    isLoading: boolean;
    output: string;
    error: boolean;
    image: string | null;
    clearOutput: () => void;
}) => (
    <Pane
        position="relative"
        borderRadius={8}
        overflow="hidden"
        marginTop={"1em"}
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

export default CodeBlock;
