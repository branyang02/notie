import "katex/dist/katex.min.css";
import "bootstrap/dist/css/bootstrap.min.css";
import styles from "../styles//Notie.module.css";
import "../styles/notie-global.css";

import React, { useRef, useState, useEffect } from "react";
import { Pane } from "evergreen-ui";
import ScrollToTopButton from "./ScrollToTopButton";
import NoteToc from "./NotieToc";
import MarkdownRenderer from "./MarkdownRenderer";
import { MarkdownProcessor } from "../utils/MarkdownProcessor";
import EquationReference from "./EquationReference";
import { createRoot } from "react-dom/client";
import { NotieConfig, NotieThemes } from "../config/NotieConfig";
import { useNotieConfig } from "../utils/useNotieConfig";

import init, { RustMarkdownProcessor } from "../../markdown_processor/pkg";

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
    const mdProcessor = new MarkdownProcessor(markdown, config);
    const { markdownContent, equationMapping } = mdProcessor.process();
    const contentRef = useRef<HTMLDivElement>(null);
    const [activeId, setActiveId] = useState<string>("");

    useEffect(() => {
        const runWasm = async () => {
            await init();

            const mp = new RustMarkdownProcessor(markdown);
            mp.process();
            console.log(mp.get_markdown_content());
            console.log(mp.get_equation_mapping());
        };

        runWasm();
    }, [markdown]);

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
    }, [config.theme.blockquoteStyle, markdownContent]);

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
                    previewEquation={config.previewEquations}
                />
            );
            createRoot(equReference).render(equReferenceComponent);
            ref.parentNode?.replaceChild(equReference, ref);
        });
    }, [config.previewEquations, equationMapping, markdownContent]);

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
                            markdownContent={markdownContent}
                            activeId={activeId}
                            config={config}
                        />
                    </Pane>
                )}
                <Pane className={styles["mw-content-container"]}>
                    <Pane className={styles["blog-content"]} ref={contentRef}>
                        <MarkdownRenderer
                            markdownContent={markdownContent}
                            config={config}
                            customComponents={customComponents}
                        />
                        <ScrollToTopButton />
                    </Pane>
                </Pane>
            </Pane>
        </Pane>
    );
};

export default Notie;
