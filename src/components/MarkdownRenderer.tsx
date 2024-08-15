import React, { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import CodeBlock from "./CodeBlock";
import TikZ from "./TikZ";
import StaticCodeBlock from "./StaticCodeBlock";
import { FullNotieConfig } from "../config/NotieConfig";

type CodeProps = React.HTMLAttributes<HTMLElement> & {
    node?: unknown;
    inline?: boolean;
    className?: string;
    children?: React.ReactNode;
};

const MarkdownRenderer: React.FC<{
    markdownContent: string;
    config: FullNotieConfig;
}> = React.memo(({ markdownContent, config }) => {
    const components = useMemo(
        () => ({
            code({ inline, className, children, ...props }: CodeProps) {
                const match = /\w+/.exec(className || "");

                if (!inline && match) {
                    const language = className?.split("language-").pop() || "";
                    const content = Array.isArray(children)
                        ? children.join("")
                        : children;
                    const code = String(content).replace(/\n$/, "");
                    if (language.includes("execute-")) {
                        return (
                            <CodeBlock
                                initialCode={code}
                                language={language.split("-").pop()}
                                theme={config.theme.liveCodeTheme}
                                copyButtonHoverColor={
                                    config.theme.codeCopyButtonHoverColor
                                }
                            />
                        );
                    }
                    if (language === "tikz") {
                        return <TikZ tikzScript={code} />;
                    }
                    return (
                        <StaticCodeBlock
                            code={code}
                            language={language}
                            theme={config.theme.staticCodeTheme}
                            copyButtonHoverColor={
                                config.theme.codeCopyButtonHoverColor
                            }
                        />
                    );
                } else {
                    return (
                        <code className={className} {...props}>
                            {children}
                        </code>
                    );
                }
            },
        }),
        [config.theme.codeCopyButtonHoverColor, config.theme.liveCodeTheme, config.theme.staticCodeTheme],
    );

    const katexOptions = {
        macros: {
            "\\eqref": "\\href{\\#pre-eqn-#1}{(#1)}",
            "\\ref": "\\href{\\#pre-eqn-#1}{#1}",
            "\\label": "\\htmlId{#1}{}",
        },
        trust: (context: { command: string }) =>
            ["\\htmlId", "\\href"].includes(context.command),
    };

    return (
        <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[[rehypeKatex, katexOptions], rehypeRaw, rehypeSlug]}
            components={components}
        >
            {markdownContent}
        </ReactMarkdown>
    );
});

export default MarkdownRenderer;
