import "katex/dist/katex.min.css";
import "bootstrap/dist/css/bootstrap.min.css";
import styles from "../styles//Notie.module.css";
import "../styles/notie-global.css";

import React, {
    useCallback,
    useRef,
    useState,
    useEffect,
    useMemo,
} from "react";
import { Pane } from "evergreen-ui";
import ScrollToTopButton from "./ScrollToTopButton";
import NoteToc from "./NotieToc";
import MarkdownRenderer from "./MarkdownRenderer";
import { MarkdownProcessor } from "../utils/MarkdownProcessor";
import { NotieConfig, NotieThemes } from "../config/NotieConfig";
import { useNotieConfig } from "../utils/useNotieConfig";
import { extractTableOfContents } from "../utils/toc";

export interface NotieProps {
    markdown: string;
    config?: NotieConfig;
    theme?: NotieThemes;
    customComponents?: {
        [key: string]: () => JSX.Element;
    };
}

const Notie: React.FC<NotieProps> = ({
    markdown,
    config: userConfig,
    theme = "default",
    customComponents,
}) => {
    const config = useNotieConfig(userConfig, theme);
    const {
        markdownContent,
        markdownSections,
        equationMapping,
        blockquoteMapping,
    } = useMemo(() => {
        const processor = new MarkdownProcessor(markdown, config);
        return processor.process();
    }, [markdown, config]);
    const tocEntries = useMemo(
        () => extractTableOfContents(markdownContent),
        [markdownContent],
    );
    const contentRef = useRef<HTMLDivElement>(null);
    const [activeId, setActiveId] = useState<string>("");
    const [renderedSectionCount, setRenderedSectionCount] = useState(0);
    const [renderAllToken, setRenderAllToken] = useState(0);
    const [pendingScrollId, setPendingScrollId] = useState<string | null>(null);
    const handleRenderedSectionsChange = useCallback(
        (count: number) => setRenderedSectionCount(count),
        [],
    );
    const requestRenderAll = useCallback(() => {
        setRenderAllToken((token) => token + 1);
    }, []);
    const handleTocNavigate = useCallback(
        (id: string, event: React.MouseEvent<HTMLAnchorElement>) => {
            if (document.getElementById(id)) return;

            event.preventDefault();
            setPendingScrollId(id);
            requestRenderAll();
        },
        [requestRenderAll],
    );

    useEffect(() => {
        if (typeof window === "undefined") return;
        const hash = window.location.hash.slice(1);
        if (!hash) return;

        const id = decodeURIComponent(hash);
        if (document.getElementById(id)) return;

        setPendingScrollId(id);
        requestRenderAll();
    }, [markdownContent, requestRenderAll]);

    useEffect(() => {
        if (!pendingScrollId || typeof window === "undefined") return;

        const target = document.getElementById(pendingScrollId);
        if (!target) return;

        target.scrollIntoView();
        if (window.location.hash !== `#${pendingScrollId}`) {
            window.history.pushState(null, "", `#${pendingScrollId}`);
        }
        setPendingScrollId(null);
    }, [pendingScrollId, renderedSectionCount]);

    // Effect to observe headings and update activeId
    useEffect(() => {
        if (!contentRef.current) return;
        const observerOptions = {
            rootMargin: "0px 0px -90% 0px",
            threshold: 0,
        };

        const headings = contentRef.current.querySelectorAll(
            "h1, h2, h3, h4, h5, h6",
        );
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    const id = entry.target.id;
                    setActiveId((current) => (current === id ? current : id));
                }
            });
        }, observerOptions);

        headings.forEach((heading) => observer.observe(heading));

        return () => observer.disconnect();
    }, [markdownContent, renderedSectionCount]);

    // Effect to auto label equation numbers
    useEffect(() => {
        if (!contentRef.current) return;
        const sections = contentRef.current.getElementsByClassName("sections");

        for (
            let sectionIndex = 0;
            sectionIndex < sections.length;
            sectionIndex++
        ) {
            const section = sections[sectionIndex];
            const eqns = section.getElementsByClassName("eqn-num");
            for (let eqnIndex = 0; eqnIndex < eqns.length; eqnIndex++) {
                const eqn = eqns[eqnIndex];
                eqn.id = `eqn-${sectionIndex + 1}.${eqnIndex + 1}`;
                eqn.textContent = `(${sectionIndex + 1}.${eqnIndex + 1})`;
            }
        }
    }, [markdownContent, renderedSectionCount]);

    // Effect to auto label Definitions, Theorems, Lemmas, only for LaTeX style
    useEffect(() => {
        if (config.theme.blockquoteStyle !== "latex") return;
        if (!contentRef.current) return;
        const sections = contentRef.current.getElementsByClassName("sections");

        for (
            let sectionIndex = 0;
            sectionIndex < sections.length;
            sectionIndex++
        ) {
            const section = sections[sectionIndex];
            const definitions = section.getElementsByClassName("definition");
            const problems = section.getElementsByClassName("problem");
            const algorithms = section.getElementsByClassName("algorithm");
            const theoremsAndLemmas =
                section.querySelectorAll(".theorem, .lemma");
            for (let defIndex = 0; defIndex < definitions.length; defIndex++) {
                const def = definitions[defIndex];
                def.setAttribute(
                    "blockquote-definition-number",
                    `Definition ${sectionIndex + 1}.${defIndex + 1}`,
                );
            }
            for (let probIndex = 0; probIndex < problems.length; probIndex++) {
                const prob = problems[probIndex];
                prob.setAttribute(
                    "blockquote-problem-number",
                    `Problem ${sectionIndex + 1}.${probIndex + 1}`,
                );
            }
            for (let algIndex = 0; algIndex < algorithms.length; algIndex++) {
                const alg = algorithms[algIndex];
                alg.setAttribute(
                    "blockquote-algorithm-number",
                    `Algorithm ${sectionIndex + 1}.${algIndex + 1}`,
                );
            }
            theoremsAndLemmas.forEach((item, index) => {
                if (item.classList.contains("theorem")) {
                    item.setAttribute(
                        "blockquote-theorem-number",
                        `Theorem ${sectionIndex + 1}.${index + 1}`,
                    );
                } else if (item.classList.contains("lemma")) {
                    item.setAttribute(
                        "blockquote-theorem-number",
                        `Lemma ${sectionIndex + 1}.${index + 1}`,
                    );
                }
            });
        }
    }, [config.theme.blockquoteStyle, markdownContent, renderedSectionCount]);

    return (
        <Pane className={styles["notie-container"]}>
            <Pane
                className={
                    config.showTableOfContents
                        ? styles["mw-page-container-inner"]
                        : ""
                }
            >
                {config.showTableOfContents && (
                    <Pane className={styles["vector-column-start"]}>
                        <NoteToc
                            tocEntries={tocEntries}
                            activeId={activeId}
                            config={config}
                            onNavigate={handleTocNavigate}
                        />
                    </Pane>
                )}
                <Pane className={styles["mw-content-container"]}>
                    <Pane className={styles["blog-content"]} ref={contentRef}>
                        <MarkdownRenderer
                            markdownContent={markdownContent}
                            markdownSections={markdownSections}
                            config={config}
                            equationMapping={equationMapping}
                            blockquoteMapping={blockquoteMapping}
                            customComponents={customComponents}
                            renderAllToken={renderAllToken}
                            onRenderedSectionsChange={
                                handleRenderedSectionsChange
                            }
                        />
                        <ScrollToTopButton />
                    </Pane>
                </Pane>
            </Pane>
        </Pane>
    );
};

export default Notie;
