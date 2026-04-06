import { Tooltip } from "evergreen-ui";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { BlockquoteMapping, extractBlockquoteInfo } from "../utils/utils";

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
    children,
    blockquoteMapping,
    previewBlockquotes,
}: {
    children: Element;
    blockquoteMapping: BlockquoteMapping;
    previewBlockquotes?: boolean;
}) => {
    const { blockquoteNumber, blockquoteType, blockquoteContent } =
        extractBlockquoteInfo(children, blockquoteMapping);

    if (blockquoteContent === "error") {
        return <span style={{ color: "red" }}>{blockquoteNumber}</span>;
    }

    const displayType =
        blockquoteType.charAt(0).toUpperCase() + blockquoteType.slice(1);
    const label = `${displayType} ${blockquoteNumber}`;
    const targetId = children.getAttribute("href")?.split("#bqref-").pop();

    return previewBlockquotes ? (
        <Tooltip
            content={
                <BlockquoteCard
                    label={label}
                    content={blockquoteContent}
                    blockquoteType={blockquoteType}
                />
            }
            appearance="card"
            statelessProps={{ maxWidth: "100%" }}
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
}: {
    label: string;
    content: string;
    blockquoteType: string;
}) => {
    const style = BLOCKQUOTE_STYLES[blockquoteType] ?? DEFAULT_STYLE;
    return (
        <div
            style={{
                background: style.background,
                borderRadius: "5px",
                paddingLeft: "0.8rem",
                paddingRight: "0.8rem",
                paddingBottom: "0.9em",
                maxWidth: "400px",
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
                rehypePlugins={[[rehypeKatex]]}
            >
                {content}
            </ReactMarkdown>
        </div>
    );
};
