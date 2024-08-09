import "katex/dist/katex.min.css";
import "bootstrap/dist/css/bootstrap.min.css";
import "../styles/notie.css";

import React, { useRef, useState, useEffect } from "react";
import { Pane } from "evergreen-ui";
import ScrollToTopButton from "./ScrollToTopButton";
import NoteToc from "./NoteToc";
import MarkdownRenderer from "./MarkdownRenderer";
import { processSection } from "../utils";

export interface NotieProps {
    markdown: string;
    darkMode?: boolean;
    style?: React.CSSProperties;
}

function preProcessMarkdown(markdownContent: string): string {
    const pattern = /^(```(\w+)|## .+)$/gm;
    const parts: string[] = [];
    const equationMapping: { [key: string]: string } = {};

    let lastIndex = 0;
    let sectionIndex = 0;
    let currentSectionContent = "";

    markdownContent.replace(pattern, (match, _p1, p2, offset) => {
        if (sectionIndex > 0) {
            currentSectionContent += markdownContent.slice(lastIndex, offset);
        } else {
            parts.push(markdownContent.slice(lastIndex, offset));
        }

        if (p2) {
            // Code block
            currentSectionContent += `\`\`\`language-${p2}`;
        } else {
            // Add section dividers
            if (sectionIndex > 0) {
                currentSectionContent += `</div>\n`;
                parts.push(
                    processSection(
                        currentSectionContent,
                        sectionIndex,
                        equationMapping,
                    ),
                );
                currentSectionContent = "";
            }
            sectionIndex++;
            currentSectionContent += `<div className="sections" id="section-${sectionIndex}">\n\n${match}\n`;
        }

        lastIndex = offset + match.length;
        return match;
    });

    currentSectionContent += markdownContent.slice(lastIndex);

    if (sectionIndex > 0) {
        currentSectionContent += "</div>\n";
        parts.push(
            processSection(
                currentSectionContent,
                sectionIndex,
                equationMapping,
            ),
        );
    } else {
        parts.push(currentSectionContent);
    }

    return parts.join("");
}

const Notie: React.FC<NotieProps> = ({ markdown, darkMode, style }) => {
    const markdownContent = preProcessMarkdown(markdown);
    const contentRef = useRef<HTMLDivElement>(null);
    const [activeId, setActiveId] = useState<string>("");

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

    return (
        <Pane background={darkMode ? "#333" : "white"} style={style}>
            <Pane className="mw-page-container-inner">
                <Pane className="vector-column-start">
                    <NoteToc
                        markdownContent={markdownContent}
                        darkMode={darkMode}
                        activeId={activeId}
                    />
                </Pane>
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
