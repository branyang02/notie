import "../styles/note-toc.css";
import { Pane } from "evergreen-ui";
import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeSlug from "rehype-slug";

const generateTableOfContents = (
    markdownContent: string,
    activeId: string,
): string => {
    let res = "# Contents\n---\n";
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
            .replace(/[+.()']/g, "")
            .replace(/&/g, "");

        const formattedTitle =
            activeId === id ? `**${cleanedTitle}**` : cleanedTitle;

        res += `${"\t".repeat(level - 2)}-  [${formattedTitle}](#${id})\n`;
    }

    return res;
};

const NoteToc = ({
    markdownContent,
    activeId,
    darkMode,
}: {
    markdownContent: string;
    activeId: string;
    darkMode: boolean;
}) => {
    const toc = useMemo(
        () => generateTableOfContents(markdownContent, activeId),
        [markdownContent, activeId],
    );

    return (
        <Pane
            position="sticky"
            top={0}
            overflowY="auto"
            maxHeight="100vh"
            className={`note-toc ${darkMode ? "dark-mode" : ""}`}
        >
            <ReactMarkdown rehypePlugins={[[rehypeSlug, { prefix: "toc-" }]]}>
                {toc}
            </ReactMarkdown>
        </Pane>
    );
};

export default NoteToc;
