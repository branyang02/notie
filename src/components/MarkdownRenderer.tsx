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
import { InlineAlert } from "evergreen-ui";

type CodeProps = React.HTMLAttributes<HTMLElement> & {
    node?: unknown;
    inline?: boolean;
    className?: string;
    children?: React.ReactNode;
};

type CustomComponentFormat = {
    componentName: string;
};

const MarkdownRenderer: React.FC<{
    markdownContent: string;
    config: FullNotieConfig;
    customComponents?: {
        [key: string]: () => JSX.Element;
    };
}> = React.memo(({ markdownContent, config, customComponents }) => {
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
                            />
                        );
                    }
                    if (language === "tikz") {
                        return <TikZ tikzScript={code} />;
                    }
                    if (language === "component") {
                        if (!customComponents) {
                            return (
                                <InlineAlert intent="danger">
                                    You need to pass `customComponents` to
                                    render component code block.
                                </InlineAlert>
                            );
                        }
                        const jsonString = code.replace(/(\w+):/g, '"$1":');
                        const componentConfig = JSON.parse(
                            jsonString,
                        ) as CustomComponentFormat;

                        const CustomComponent =
                            customComponents[componentConfig.componentName];

                        if (CustomComponent) {
                            return <CustomComponent />;
                        } else {
                            return (
                                <InlineAlert intent="danger">
                                    {`We couldn't find your component \`${componentConfig.componentName}\`.`}
                                </InlineAlert>
                            );
                        }
                    }
                    return (
                        <StaticCodeBlock
                            code={code}
                            language={language}
                            theme={config.theme.staticCodeTheme}
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
        [
            config.theme.liveCodeTheme,
            config.theme.staticCodeTheme,
            customComponents,
        ],
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
