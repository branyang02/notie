import { Tooltip } from "evergreen-ui";
import React, { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { katexOptions } from "../utils/katexOptions";
import rehypeAccessibleKatexRefs from "../utils/rehypeAccessibleKatexRefs";
import { sanitizeUrl } from "../utils/sanitizeUrl";
import {
    BlockquoteMapping,
    EquationMapping,
    extractBlockquoteInfo,
} from "../utils/utils";
import EquationReference from "./EquationReference";

interface BlockquoteStyle {
    background: string;
    labelColor: string;
}

// The preview card shares the theme-aware CSS variables that style
// blockquotes in notie-global.css (set per appearance in useNotieConfig).
// Lemmas share the theorem colors, mirroring the blockquote CSS rules.
const blockquoteStyleFor = (blockquoteType: string): BlockquoteStyle => {
    const knownTypes = [
        "definition",
        "proof",
        "equation",
        "theorem",
        "lemma",
        "algorithm",
        "problem",
        "important",
        "note",
    ];
    let variableType = knownTypes.includes(blockquoteType)
        ? blockquoteType
        : "theorem";
    if (variableType === "lemma") variableType = "theorem";

    return {
        background: `var(--blog-bq-${variableType}-bg, rgba(126, 174, 247, 0.2))`,
        labelColor: `var(--blog-bq-${variableType}-label, #486bd5)`,
    };
};

const BlockquoteReference = ({
    href,
    blockquoteMapping,
    equationMapping,
    previewBlockquotes,
    previewEquations,
}: {
    href: string;
    blockquoteMapping: BlockquoteMapping;
    equationMapping: EquationMapping;
    previewBlockquotes?: boolean;
    previewEquations?: boolean;
}) => {
    const { blockquoteNumber, blockquoteType, blockquoteContent } =
        extractBlockquoteInfo(href, blockquoteMapping);

    if (blockquoteContent === "error") {
        return (
            <span style={{ color: "red" }}>
                {/* Screen-reader prefix so the error state is not conveyed
                    by the red color alone. */}
                <span className="sr-only">Unresolved reference: </span>
                {blockquoteNumber}
            </span>
        );
    }

    const displayType =
        blockquoteType.charAt(0).toUpperCase() + blockquoteType.slice(1);
    const label = `${displayType} ${blockquoteNumber}`;
    const targetId = href.split("#bqref-").pop();

    return previewBlockquotes ? (
        <Tooltip
            content={
                <BlockquoteCard
                    label={label}
                    content={blockquoteContent}
                    blockquoteType={blockquoteType}
                    blockquoteMapping={blockquoteMapping}
                    equationMapping={equationMapping}
                    previewEquations={previewEquations}
                />
            }
            appearance="card"
            statelessProps={{ maxWidth: "100%", paddingX: 8, paddingY: 8 }}
        >
            <a href={`#${targetId}`} aria-label={label}>
                <span>{label}</span>
            </a>
        </Tooltip>
    ) : (
        <a href={`#${targetId}`} aria-label={label}>
            <span>{label}</span>
        </a>
    );
};

export default BlockquoteReference;

const BlockquoteCard = ({
    label,
    content,
    blockquoteType,
    blockquoteMapping,
    equationMapping,
    previewEquations,
}: {
    label: string;
    content: string;
    blockquoteType: string;
    blockquoteMapping: BlockquoteMapping;
    equationMapping: EquationMapping;
    previewEquations?: boolean;
}) => {
    const style = blockquoteStyleFor(blockquoteType);
    const strippedContent = content.replace(/\\label\{[^}]*\}/g, "");
    const components = useMemo(
        () => ({
            a({
                href = "",
                children,
                ...props
            }: React.AnchorHTMLAttributes<HTMLAnchorElement> & {
                node?: unknown;
                children?: React.ReactNode;
            }) {
                if (href.startsWith("#pre-eqn-")) {
                    return (
                        <EquationReference
                            href={href}
                            textContent={textFromReactNode(children)}
                            equationMapping={equationMapping}
                            previewEquation={previewEquations}
                            inert={
                                "data-notie-inert-ref" in
                                (props as Record<string, unknown>)
                            }
                        />
                    );
                }

                if (href.startsWith("#bqref-")) {
                    // Render nested blockquote references as styled links
                    // without their own preview so tooltips never nest
                    // (and never recurse) inside a preview card.
                    return (
                        <BlockquoteReference
                            href={href}
                            blockquoteMapping={blockquoteMapping}
                            equationMapping={equationMapping}
                            previewBlockquotes={false}
                            previewEquations={previewEquations}
                        />
                    );
                }

                return (
                    <a href={href} {...props}>
                        {children}
                    </a>
                );
            },
        }),
        [blockquoteMapping, equationMapping, previewEquations],
    );

    return (
        <div
            style={{
                // Layer the translucent type tint over the theme background
                // so the card matches in-page blockquotes in both light and
                // dark appearances (the tooltip surface color never shows
                // through).
                backgroundColor: "var(--blog-background-color)",
                backgroundImage: `linear-gradient(${style.background}, ${style.background})`,
                color: "var(--blog-text-color)",
                borderRadius: "5px",
                paddingLeft: "0.8rem",
                paddingRight: "0.8rem",
                paddingBottom: "0.9em",
                maxWidth: "800px",
                fontFamily: "var(--blog-font-family)",
                fontSize: "var(--blog-font-size)",
            }}
        >
            <span
                style={{
                    display: "block",
                    fontWeight: "bold",
                    textTransform: "uppercase",
                    fontSize: "0.85em",
                    paddingTop: "0.9rem",
                    color: style.labelColor,
                }}
            >
                {label}.
            </span>
            <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[
                    [rehypeKatex, katexOptions],
                    rehypeAccessibleKatexRefs,
                ]}
                components={components}
                urlTransform={sanitizeUrl}
            >
                {strippedContent}
            </ReactMarkdown>
        </div>
    );
};

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
