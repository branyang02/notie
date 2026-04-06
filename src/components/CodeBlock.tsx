import { indentUnit } from "@codemirror/language";
import { Extension } from "@codemirror/state";
import { EditorView } from "@codemirror/view";
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
import { useCallback, useEffect, useRef, useState } from "react";
import type { BundledTheme } from "shiki";
import shiki from "codemirror-shiki";

import { runCode, RunCodeResponse } from "../service/api";
import CodeHeader from "./CodeHeader";
import styles from "../styles/Notie.module.css";
import { getHighlighter, resolveLanguage } from "../utils/shikiHighlighter";

const CodeBlock = ({
    initialCode,
    language = "python",
    theme,
}: {
    initialCode: string;
    language?: string;
    theme: string;
}) => {
    const [extensions, setExtensions] = useState<Extension[]>([
        indentUnit.of("    "),
    ]);

    useEffect(() => {
        let cancelled = false;
        getHighlighter().then((highlighter) => {
            if (cancelled) return;
            const resolvedTheme = highlighter.getTheme(theme as BundledTheme);
            const { bg, fg, type, colors } = resolvedTheme;
            const selectionBg =
                colors?.["editor.selectionBackground"] ??
                (type === "dark"
                    ? "rgba(255,255,255,0.15)"
                    : "rgba(0,0,0,0.1)");
            setExtensions([
                shiki({
                    highlighter: Promise.resolve(highlighter),
                    language: resolveLanguage(language),
                    theme: theme as BundledTheme,
                }),
                EditorView.theme({
                    "&": { backgroundColor: bg },
                    ".cm-gutters": { backgroundColor: bg, borderRight: "none" },
                    "&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection":
                        { backgroundColor: selectionBg },
                    "& .cm-cursor, & .cm-dropCursor": { borderLeftColor: fg },
                }),
                indentUnit.of("    "),
            ]);
        });
        return () => {
            cancelled = true;
        };
    }, [language, theme]);

    const [isLoading, setIsLoading] = useState(false);
    const [output, setOutput] = useState("");
    const [error, setError] = useState(false);
    const [code, setCode] = useState(initialCode);
    const [image, setImage] = useState("");
    const editorRef = useRef<ReactCodeMirrorRef>(null);

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
            <CodeHeader language={language} code={code} />
            <Pane>
                <Pane
                    position="relative"
                    overflow="hidden"
                    style={{ borderRadius: "0 0 10px 10px" }}
                >
                    <div className={styles["code-blocks"]}>
                        <CodeMirror
                            ref={editorRef}
                            value={initialCode}
                            extensions={extensions}
                            maxHeight="80vh"
                            minHeight="100px"
                            theme="none"
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
