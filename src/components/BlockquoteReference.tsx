import { Tooltip } from "evergreen-ui";
import React, { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { katexOptions } from "../utils/katexOptions";
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

const BLOCKQUOTE_STYLES: Record<string, BlockquoteStyle> = {
    definition: {
        background: "rgba(174, 247, 126, 0.2)",
        labelColor: "#31dd2e",
    },
    proof: { background: "rgba(174, 247, 126, 0.2)", labelColor: "#31dd2e" },
    theorem: { background: "rgba(126, 174, 247, 0.2)", labelColor: "#486bd5" },
    lemma: { background: "rgba(126, 174, 247, 0.2)", labelColor: "#486bd5" },
    algorithm: {
        background: "rgba(126, 174, 247, 0.2)",
        labelColor: "#486bd5",
    },
    problem: { background: "rgba(126, 174, 247, 0.2)", labelColor: "#486bd5" },
    important: {
        background: "rgba(247, 126, 126, 0.2)",
        labelColor: "#dd2e2e",
    },
    note: {
        background: "rgb(255 253 0 / 19%)",
        labelColor: "lch(86 109.24 91.22)",
    },
};

const DEFAULT_STYLE: BlockquoteStyle = {
    background: "rgba(126, 174, 247, 0.2)",
    labelColor: "#486bd5",
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
        return <span style={{ color: "red" }}>{blockquoteNumber}</span>;
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
                    equationMapping={equationMapping}
                    previewEquations={previewEquations}
                />
            }
            appearance="card"
            statelessProps={{ maxWidth: "100%", paddingX: 8, paddingY: 8 }}
        >
            <a href={`#${targetId}`}>
                <span>{label}</span>
            </a>
        </Tooltip>
    ) : (
        <a href={`#${targetId}`}>
            <span>{label}</span>
        </a>
    );
};

export default BlockquoteReference;

const BlockquoteCard = ({
    label,
    content,
    blockquoteType,
    equationMapping,
    previewEquations,
}: {
    label: string;
    content: string;
    blockquoteType: string;
    equationMapping: EquationMapping;
    previewEquations?: boolean;
}) => {
    const style = BLOCKQUOTE_STYLES[blockquoteType] ?? DEFAULT_STYLE;
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
        [equationMapping, previewEquations],
    );

    return (
        <div
            style={{
                background: style.background,
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
                rehypePlugins={[[rehypeKatex, katexOptions]]}
                components={components}
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
