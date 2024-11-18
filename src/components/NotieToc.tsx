import { Pane } from "evergreen-ui";
import rehypeRaw from "rehype-raw";
import { useEffect, useMemo, useRef } from "react";
import ReactMarkdown from "react-markdown";
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
            .replace(/&nbsp;/g, "")
            .replace(/&/g, "")
            .replace(/:/g, "");

        const formattedTitle =
            activeId === id ? `**${cleanedTitle}**` : cleanedTitle;

        res += `${"\t".repeat(level - 2)}- <a id="toc-${id}" href="#${id}">${formattedTitle}</a>\n`;
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
    const tocRef = useRef<HTMLDivElement>(null); // Ref for the TOC container

    const toc = useMemo(
        () =>
            generateTableOfContents(markdownContent, activeId, config.tocTitle),
        [markdownContent, activeId, config.tocTitle],
    );

    useEffect(() => {
        const tocContainer = tocRef.current;
        if (tocContainer) {
            const activeElement = tocContainer.querySelector(
                `#toc-${activeId}`,
            );
            if (activeElement) {
                const tocRect = tocContainer.getBoundingClientRect();
                const activeRect = activeElement.getBoundingClientRect();
                const offset =
                    activeRect.top -
                    tocRect.top -
                    tocContainer.clientHeight / 2;

                tocContainer.scrollTo({
                    top: tocContainer.scrollTop + offset,
                    behavior: "smooth",
                });
            }
        }
    }, [activeId]);

    return (
        <Pane
            position="sticky"
            top={0}
            overflowY="auto"
            maxHeight="100vh"
            className={styles["note-toc"]}
            ref={tocRef}
        >
            <ReactMarkdown rehypePlugins={[rehypeRaw]}>{toc}</ReactMarkdown>
        </Pane>
    );
};

export default NotieToc;
