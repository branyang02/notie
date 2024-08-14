import "katex/dist/katex.min.css";
import "bootstrap/dist/css/bootstrap.min.css";
import "../styles/notie.css";

import React, { useRef, useState, useEffect } from "react";
import { Pane } from "evergreen-ui";
import ScrollToTopButton from "./ScrollToTopButton";
import NoteToc from "./NoteToc";
import MarkdownRenderer from "./MarkdownRenderer";
import { MarkdownProcessor } from "../utils/MarkdownProcessor";
import EquationReference from "./EquationReference";
import { createRoot } from "react-dom/client";
import { NotieConfig } from "../config/NotieConfig";
import { useNotieConfig } from "../utils/useNotieConfig";

export interface NotieProps {
    markdown: string;
    darkMode?: boolean;
    previewEquation?: boolean;
    config?: NotieConfig;
}

const Notie: React.FC<NotieProps> = ({
    markdown,
    darkMode = false,
    previewEquation = true,
    config = {},
}) => {
    const mdProcessor = new MarkdownProcessor(markdown);
    const { markdownContent, equationMapping } = mdProcessor.process();
    const contentRef = useRef<HTMLDivElement>(null);
    const [activeId, setActiveId] = useState<string>("");

    const mergedConfig = useNotieConfig(config);

    // Effect to observe headings and update activeId
    useEffect(() => {
        const observerOptions = {
            rootMargin: "0px 0px -80% 0px",
            threshold: 1.0,
        };

        if (!contentRef.current) return;

        const headings = contentRef.current.querySelectorAll(
            "h1, h2, h3, h4, h5, h6",
        );
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    setActiveId(entry.target.id);
                }
            });
        }, observerOptions);

        headings.forEach((heading) => observer.observe(heading));

        return () => {
            headings.forEach((heading) => observer.unobserve(heading));
        };
    }, [markdownContent]);

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
    }, [markdownContent]);

    // Effect to enable Equation Preview
    useEffect(() => {
        if (!contentRef.current) return;

        const eqnRefs = contentRef.current.querySelectorAll(
            'a[href^="#pre-eqn-"]',
        );

        eqnRefs.forEach((ref) => {
            const equReference = document.createElement("span");
            const equReferenceComponent = (
                <EquationReference
                    children={ref}
                    equationMapping={equationMapping}
                    previewEquation={previewEquation}
                />
            );
            createRoot(equReference).render(equReferenceComponent);
            ref.parentNode?.replaceChild(equReference, ref);
        });
    }, [equationMapping, markdownContent, previewEquation]);

    return (
        <Pane background={darkMode ? "#333" : "white"}>
            <Pane
                className={
                    mergedConfig.showTableOfContents
                        ? "mw-page-container-inner"
                        : ""
                }
            >
                {mergedConfig.showTableOfContents && (
                    <Pane className="vector-column-start">
                        <NoteToc
                            markdownContent={markdownContent}
                            darkMode={darkMode}
                            activeId={activeId}
                            config={mergedConfig}
                        />
                    </Pane>
                )}
                <Pane className="mw-content-container">
                    <Pane
                        className={`blog-content ${darkMode ? "dark-mode" : ""}`}
                        ref={contentRef}
                    >
                        <MarkdownRenderer
                            markdownContent={markdownContent}
                            darkMode={darkMode}
                        />
                        <ScrollToTopButton />
                    </Pane>
                </Pane>
            </Pane>
        </Pane>
    );
};

export default Notie;
