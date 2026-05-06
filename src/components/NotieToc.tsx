import { Pane } from "evergreen-ui";
import React, { useEffect, useRef } from "react";
import { NotieConfig } from "../config/NotieConfig";
import styles from "../styles/NotieToc.module.css";
import { TocEntry } from "../utils/toc";

const NotieToc = ({
    tocEntries,
    activeId,
    config,
    onNavigate,
}: {
    tocEntries: TocEntry[];
    activeId: string;
    config: NotieConfig;
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
            behavior: "smooth",
        });
    }, [activeId]);

    return (
        <Pane
            is="nav"
            aria-label={config.tocTitle}
            position="sticky"
            top={0}
            overflowY="auto"
            maxHeight="100vh"
            className={styles["note-toc"]}
            ref={tocRef}
        >
            <h1>{config.tocTitle}</h1>
            <hr />
            <ul>
                {tocEntries.map((entry, index) => (
                    <li
                        key={`${entry.id}-${entry.level}-${index}`}
                        style={{ marginLeft: `${entry.level - 2}em` }}
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

export default NotieToc;
