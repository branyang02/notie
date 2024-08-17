import { Pane } from "evergreen-ui";
import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeSlug from "rehype-slug";
import { NotieConfig } from "../config/NotieConfig";
import styles from "../styles/NotieToc.module.css";

const generateTableOfContents = (
    markdownContent: string,
    activeId: string,
    tocTitle?: string,
): string => {
    let res = `# ${tocTitle}\n---\n`;
    const pattern = /^#+ (.*)$/gm;
    let match;

    while ((match = pattern.exec(markdownContent)) !== null) {
        const [fullMatch, title] = match;
        const level = fullMatch.match(/^#+/)?.[0].length || 0;
        if (level === 1) continue;

        const cleanedTitle = title.replace(/[*]/g, "").trim();
        const id = cleanedTitle
            .replace(/\s+/g, "-")
            .toLowerCase()
            .replace(/[+.()'`]/g, "")
            .replace(/&/g, "");

        const formattedTitle =
            activeId === id ? `**${cleanedTitle}**` : cleanedTitle;

        res += `${"\t".repeat(level - 2)}-  [${formattedTitle}](#${id})\n`;
    }

    return res;
};

const NotieToc = ({
    markdownContent,
    activeId,
    config,
}: {
    markdownContent: string;
    activeId: string;
    config: NotieConfig;
}) => {
    const toc = useMemo(
        () =>
            generateTableOfContents(markdownContent, activeId, config.tocTitle),
        [markdownContent, activeId, config.tocTitle],
    );

    return (
        <Pane
            position="sticky"
            top={0}
            overflowY="auto"
            maxHeight="100vh"
            className={styles["note-toc"]}
        >
            <ReactMarkdown rehypePlugins={[[rehypeSlug, { prefix: "toc-" }]]}>
                {toc}
            </ReactMarkdown>
        </Pane>
    );
};

export default NotieToc;
