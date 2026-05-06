import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { katexOptions } from "../utils/katexOptions";
import { BlockquoteMapping, EquationMapping } from "../utils/utils";
import BlockquoteReference from "./BlockquoteReference";
import CodeBlock from "./CodeBlock";
import DesmosGraph from "./DesmosGraph";
import EquationReference from "./EquationReference";
import { InlineAlert } from "evergreen-ui";
import LazyRender from "./LazyRender";
import StaticCodeBlock from "./StaticCodeBlock";
import TikZ from "./TikZ";
import { FullNotieConfig } from "../config/NotieConfig";

type CodeProps = React.HTMLAttributes<HTMLElement> & {
    node?: unknown;
    inline?: boolean;
    className?: string;
    children?: React.ReactNode;
};

type CustomComponentFormat = {
    componentName: string;
};

type AnchorProps = React.AnchorHTMLAttributes<HTMLAnchorElement> & {
    node?: unknown;
    children?: React.ReactNode;
};

const INITIAL_SECTION_COUNT = 2;

function textFromReactNode(node: React.ReactNode): string {
    if (typeof node === "string" || typeof node === "number") {
        return String(node);
    }
    if (Array.isArray(node)) {
        return node.map(textFromReactNode).join("");
    }
    if (React.isValidElement<{ children?: React.ReactNode }>(node)) {
        return textFromReactNode(node.props.children);
    }
    return "";
}

function getInitialSectionCount(sectionCount: number): number {
    if (typeof window === "undefined") return sectionCount;
    return Math.min(INITIAL_SECTION_COUNT, sectionCount);
}

function getDocumentSignature(sections: string[]): string {
    const headings: string[] = [];

    for (const section of sections) {
        const matches = section.matchAll(/^#{1,6}\s+(.+)$/gm);
        for (const match of matches) {
            headings.push(match[1].trim());
            if (headings.length >= 3) {
                return headings.join("\n");
            }
        }
    }

    return sections[0]?.slice(0, 120) ?? "";
}

function scheduleNextSection(callback: () => void): () => void {
    if (typeof window === "undefined") return () => {};

    if (typeof window.requestIdleCallback === "function") {
        const idleId = window.requestIdleCallback(callback, { timeout: 250 });
        return () => window.cancelIdleCallback(idleId);
    }

    const timeoutId = globalThis.setTimeout(callback, 16);
    return () => globalThis.clearTimeout(timeoutId);
}

const MarkdownRenderer: React.FC<{
    markdownContent: string;
    markdownSections?: string[];
    config: FullNotieConfig;
    equationMapping: EquationMapping;
    blockquoteMapping: BlockquoteMapping;
    customComponents?: {
        [key: string]: () => JSX.Element;
    };
    renderAllToken?: number;
    onRenderedSectionsChange?: (renderedSectionCount: number) => void;
}> = React.memo(
    ({
        markdownContent,
        markdownSections,
        config,
        equationMapping,
        blockquoteMapping,
        customComponents,
        renderAllToken,
        onRenderedSectionsChange,
    }) => {
        const sections = useMemo(
            () =>
                markdownSections && markdownSections.length > 0
                    ? markdownSections
                    : [markdownContent],
            [markdownContent, markdownSections],
        );
        const [visibleSectionCount, setVisibleSectionCount] = useState(() =>
            getInitialSectionCount(sections.length),
        );
        const documentSignature = useMemo(
            () => getDocumentSignature(sections),
            [sections],
        );
        const previousSectionsRef = useRef(sections);
        const previousDocumentSignatureRef = useRef(documentSignature);
        const previousRenderAllTokenRef = useRef(renderAllToken);
        const sectionsChanged = previousSectionsRef.current !== sections;
        const sameDocument =
            previousDocumentSignatureRef.current === documentSignature;
        const initialSectionCount = getInitialSectionCount(sections.length);
        const targetVisibleSectionCount =
            sectionsChanged && !sameDocument
                ? initialSectionCount
                : Math.max(visibleSectionCount, initialSectionCount);
        const effectiveVisibleSectionCount = Math.min(
            targetVisibleSectionCount,
            sections.length,
        );

        useEffect(() => {
            if (!sectionsChanged) return;
            const shouldPreserveRenderedSections =
                previousDocumentSignatureRef.current === documentSignature;

            previousSectionsRef.current = sections;
            previousDocumentSignatureRef.current = documentSignature;
            setVisibleSectionCount((current) =>
                shouldPreserveRenderedSections
                    ? Math.min(
                          Math.max(current, initialSectionCount),
                          sections.length,
                      )
                    : initialSectionCount,
            );
        }, [documentSignature, initialSectionCount, sections, sectionsChanged]);

        useEffect(() => {
            if (previousRenderAllTokenRef.current === renderAllToken) return;
            previousRenderAllTokenRef.current = renderAllToken;
            setVisibleSectionCount(sections.length);
        }, [renderAllToken, sections.length]);

        useEffect(() => {
            onRenderedSectionsChange?.(effectiveVisibleSectionCount);
        }, [effectiveVisibleSectionCount, onRenderedSectionsChange]);

        useEffect(() => {
            if (
                sectionsChanged ||
                effectiveVisibleSectionCount >= sections.length
            ) {
                return;
            }

            return scheduleNextSection(() => {
                setVisibleSectionCount((current) =>
                    Math.min(current + 1, sections.length),
                );
            });
        }, [effectiveVisibleSectionCount, sections.length, sectionsChanged]);

        const components = useMemo(
            () => ({
                a({ href = "", children, ...props }: AnchorProps) {
                    if (href.startsWith("#pre-eqn-")) {
                        return (
                            <EquationReference
                                href={href}
                                textContent={textFromReactNode(children)}
                                equationMapping={equationMapping}
                                previewEquation={config.previewEquations}
                            />
                        );
                    }

                    if (href.startsWith("#bqref-")) {
                        return (
                            <BlockquoteReference
                                href={href}
                                blockquoteMapping={blockquoteMapping}
                                equationMapping={equationMapping}
                                previewBlockquotes={config.previewBlockquotes}
                                previewEquations={config.previewEquations}
                            />
                        );
                    }

                    return (
                        <a href={href} {...props}>
                            {children}
                        </a>
                    );
                },
                code({ inline, className, children, ...props }: CodeProps) {
                    const match = /\w+/.exec(className || "");

                    if (!inline && match) {
                        const language =
                            className?.split("language-").pop() || "";
                        const content = Array.isArray(children)
                            ? children.join("")
                            : children;
                        const code = String(content).replace(/\n$/, "");
                        if (language.includes("execute-")) {
                            return (
                                <LazyRender minHeight={260}>
                                    <CodeBlock
                                        initialCode={code}
                                        language={language.split("-").pop()}
                                        theme={config.theme.liveCodeTheme}
                                    />
                                </LazyRender>
                            );
                        }
                        if (language === "tikz") {
                            return (
                                <LazyRender minHeight={220}>
                                    <TikZ tikzScript={code} />
                                </LazyRender>
                            );
                        }
                        if (language === "desmos") {
                            return (
                                <LazyRender minHeight={400}>
                                    <DesmosGraph graphScript={code} />
                                </LazyRender>
                            );
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
                blockquoteMapping,
                config.previewBlockquotes,
                config.previewEquations,
                config.theme.liveCodeTheme,
                config.theme.staticCodeTheme,
                customComponents,
                equationMapping,
            ],
        );

        return (
            <>
                {sections
                    .slice(0, effectiveVisibleSectionCount)
                    .map((section, index) => (
                        <MarkdownSection
                            key={index}
                            markdownContent={section}
                            components={components}
                        />
                    ))}
            </>
        );
    },
);

const MarkdownSection = React.memo(
    ({
        markdownContent,
        components,
    }: {
        markdownContent: string;
        components: React.ComponentProps<typeof ReactMarkdown>["components"];
    }) => (
        <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[[rehypeKatex, katexOptions], rehypeRaw, rehypeSlug]}
            components={components}
        >
            {markdownContent}
        </ReactMarkdown>
    ),
);

export default MarkdownRenderer;
