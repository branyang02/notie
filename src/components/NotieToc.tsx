import { Pane } from "evergreen-ui";
import React, { memo, useEffect, useRef } from "react";
import styles from "../styles/NotieToc.module.css";
import { TocEntry } from "../utils/toc";
import { preferredScrollBehavior } from "../utils/motion";

// Per-level indentation styles are cached so list entries reuse the same
// style object identity across renders instead of allocating a new object
// per entry on every render.
const entryIndentStyles = new Map<number, React.CSSProperties>();

function getEntryIndentStyle(level: number): React.CSSProperties {
    let style = entryIndentStyles.get(level);
    if (!style) {
        style = { marginLeft: `${level - 2}em` };
        entryIndentStyles.set(level, style);
    }
    return style;
}

const NotieToc = ({
    tocEntries,
    activeId,
    tocTitle,
    onNavigate,
}: {
    tocEntries: TocEntry[];
    activeId: string;
    tocTitle: string;
    onNavigate?: (
        id: string,
        event: React.MouseEvent<HTMLAnchorElement>,
    ) => void;
}) => {
    const tocRef = useRef<HTMLElement>(null);

    useEffect(() => {
        const tocContainer = tocRef.current;
        if (!tocContainer || !activeId) return;

        const activeElement = document.getElementById(`toc-${activeId}`);
        if (!activeElement || !tocContainer.contains(activeElement)) return;

        const tocRect = tocContainer.getBoundingClientRect();
        const activeRect = activeElement.getBoundingClientRect();
        const offset =
            activeRect.top - tocRect.top - tocContainer.clientHeight / 2;

        tocContainer.scrollTo({
            top: tocContainer.scrollTop + offset,
            behavior: preferredScrollBehavior(),
        });
    }, [activeId]);

    return (
        <Pane
            is="nav"
            aria-label={tocTitle}
            position="sticky"
            top={0}
            overflowY="auto"
            maxHeight="100vh"
            className={styles["note-toc"]}
            ref={tocRef}
        >
            <h2>{tocTitle}</h2>
            <hr />
            <ul>
                {tocEntries.map((entry, index) => (
                    <li
                        key={`${entry.id}-${entry.level}-${index}`}
                        style={getEntryIndentStyle(entry.level)}
                    >
                        <a
                            id={`toc-${entry.id}`}
                            href={`#${entry.id}`}
                            onClick={(event) => onNavigate?.(entry.id, event)}
                            className={
                                activeId === entry.id
                                    ? styles["active"]
                                    : undefined
                            }
                        >
                            {entry.title}
                        </a>
                    </li>
                ))}
            </ul>
        </Pane>
    );
};

export default memo(NotieToc);
