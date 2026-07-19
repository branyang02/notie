import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { type ExtraProps } from "react-markdown";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { katexOptions } from "../utils/katexOptions";
import { sanitizeUrl } from "../utils/sanitizeUrl";
import {
    BlockquoteMapping,
    EquationMapping,
    parseExecuteLanguage,
} from "../utils/utils";
import BlockquoteReference from "./BlockquoteReference";
import CodeBlock from "./CodeBlock";
import DesmosGraph from "./DesmosGraph";
import EquationReference from "./EquationReference";
import { InlineAlert } from "evergreen-ui";
import LazyRender from "./LazyRender";
import StaticCodeBlock from "./StaticCodeBlock";
import TikZ from "./TikZ";
import { CustomComponents, FullNotieConfig } from "../config/NotieConfig";
import type { Element as HastElement, ElementContent } from "hast";

type PreProps = React.HTMLAttributes<HTMLPreElement> &
    ExtraProps & {
        children?: React.ReactNode;
    };

type CustomComponentFormat = {
    componentName: string;
} & Record<string, unknown>;

type AnchorProps = React.AnchorHTMLAttributes<HTMLAnchorElement> & {
    node?: unknown;
    children?: React.ReactNode;
};

/**
 * Returns the sole `<code>` element child of a `<pre>` hast node, or `null`
 * when the `<pre>` does not look like a markdown code block (react-markdown
 * always renders a fenced or indented code block as a `<pre>` wrapping
 * exactly one `<code>` element).
 */
function getCodeChild(node: HastElement | undefined): HastElement | null {
    if (!node) return null;
    const meaningfulChildren = node.children.filter(
        (child) =>
            !(child.type === "text" && child.value.trim() === "") &&
            child.type !== "comment",
    );
    if (meaningfulChildren.length !== 1) return null;
    const [child] = meaningfulChildren;
    if (child.type !== "element" || child.tagName !== "code") return null;
    return child;
}

/** Extracts the `language-*` token from a hast `<code>` element, if any. */
function getCodeLanguage(codeNode: HastElement): string {
    const className = codeNode.properties?.className;
    const classes = Array.isArray(className)
        ? className.map(String)
        : typeof className === "string"
          ? className.split(/\s+/)
          : [];
    const languageClass = classes.find((cls) => cls.startsWith("language-"));
    return languageClass ? languageClass.slice("language-".length) : "";
}

/** Concatenates the text content of a hast subtree. */
function textFromHast(nodes: ElementContent[]): string {
    let text = "";
    for (const node of nodes) {
        if (node.type === "text") {
            text += node.value;
        } else if (node.type === "element") {
            text += textFromHast(node.children);
        }
    }
    return text;
}

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

/**
 * Returns the number of sections rendered on the first render pass.
 *
 * This intentionally returns the same value on the server and on the client
 * so that server-rendered markup matches the first client render during
 * hydration. Previously the server rendered every section while the client
 * started with `INITIAL_SECTION_COUNT`, which guaranteed a hydration
 * mismatch. The trade-off is that SSR output now only contains the initial
 * sections; the remaining sections are revealed progressively after
 * hydration, exactly as on a client-only render.
 */
function getInitialSectionCount(sectionCount: number): number {
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
    customComponents?: CustomComponents;
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
                // react-markdown v9 removed the `inline` prop from the
                // `code` component. The v9-idiomatic way to detect block
                // code is structural: every fenced/indented code block is
                // rendered as a `<pre>` wrapping exactly one `<code>`,
                // while inline code is a bare `<code>` (never inside a
                // custom `pre`). Implementing `pre` therefore captures all
                // block code — including classless fences (``` with no info
                // string), which have no `language-*` class and previously
                // fell through to the plain inline branch.
                pre({ node, children, ...props }: PreProps) {
                    const codeNode = getCodeChild(node);

                    if (codeNode) {
                        const language = getCodeLanguage(codeNode);
                        const code = textFromHast(codeNode.children).replace(
                            /\n$/,
                            "",
                        );
                        if (language.includes("execute-")) {
                            return (
                                <LazyRender minHeight={260}>
                                    <CodeBlock
                                        initialCode={code}
                                        language={parseExecuteLanguage(
                                            language,
                                        )}
                                        theme={config.theme.liveCodeTheme}
                                        codeRunnerUrl={config.codeRunnerUrl}
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
                                    <DesmosGraph
                                        graphScript={code}
                                        appearance={config.theme.appearance}
                                        apiKey={config.desmosApiKey}
                                    />
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
                            let componentConfig: CustomComponentFormat;
                            try {
                                const jsonString = code.replace(
                                    /(\w+):/g,
                                    '"$1":',
                                );
                                const parsed: unknown = JSON.parse(jsonString);
                                if (
                                    typeof parsed !== "object" ||
                                    parsed === null ||
                                    typeof (parsed as CustomComponentFormat)
                                        .componentName !== "string"
                                ) {
                                    throw new Error(
                                        "Missing componentName in component configuration.",
                                    );
                                }
                                componentConfig =
                                    parsed as CustomComponentFormat;
                            } catch {
                                return (
                                    <InlineAlert intent="danger">
                                        Invalid component configuration.
                                    </InlineAlert>
                                );
                            }

                            const CustomComponent =
                                customComponents[componentConfig.componentName];

                            if (CustomComponent) {
                                return (
                                    <CustomComponent config={componentConfig} />
                                );
                            } else {
                                return (
                                    <InlineAlert intent="danger">
                                        {`We couldn't find your component \`${componentConfig.componentName}\`.`}
                                    </InlineAlert>
                                );
                            }
                        }
                        // Classless fences reach here with language "";
                        // resolveLanguage maps unknown languages to "text",
                        // so they get the same StaticCodeBlock chrome as
                        // tagged fences.
                        return (
                            <StaticCodeBlock
                                code={code}
                                language={language}
                                theme={config.theme.staticCodeTheme}
                            />
                        );
                    }

                    // Not a markdown code block (e.g. raw <pre> HTML passed
                    // through rehype-raw): render the <pre> untouched.
                    // Inline code never hits this component — it renders via
                    // the default `code` handling as a plain <code>.
                    return <pre {...props}>{children}</pre>;
                },
            }),
            [
                blockquoteMapping,
                config.codeRunnerUrl,
                config.desmosApiKey,
                config.previewBlockquotes,
                config.previewEquations,
                config.theme.appearance,
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
            urlTransform={sanitizeUrl}
        >
            {markdownContent}
        </ReactMarkdown>
    ),
);

export default MarkdownRenderer;
