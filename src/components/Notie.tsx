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
import {
    CustomComponents,
    NotieConfig,
    NotieThemes,
} from "../config/NotieConfig";
import { useNotieConfig } from "../utils/useNotieConfig";
import { useShallowStableObject } from "../utils/useShallowStableObject";
import { extractTableOfContents } from "../utils/toc";

export interface NotieProps {
    markdown: string;
    config?: NotieConfig;
    theme?: NotieThemes;
    customComponents?: CustomComponents;
}

/**
 * Trailing delay applied to full-document DOM rescans while sections are
 * being progressively revealed. Must stay short (<= 100ms) so equation and
 * blockquote numbers are never visibly stale after the last reveal.
 */
const REVEAL_RESCAN_DELAY_MS = 50;

const NEVER_RAN = Symbol("never-ran");

/**
 * Runs `effect` like `useEffect`, but coalesces bursts of
 * `renderedSectionCount` changes caused by progressive section reveal.
 *
 * `MarkdownRenderer` reveals sections one at a time (one idle callback per
 * section), and the heading observer plus equation/blockquote numbering
 * effects each rescan the whole document on every reveal — O(N^2) work over
 * the reveal sequence for large documents. This hook instead:
 *
 * - runs `effect` synchronously whenever `syncKey` changes (including on
 *   mount), so numbering and observers are correct immediately for the
 *   initially rendered content and after document updates;
 * - debounces `renderedSectionCount` changes with a short trailing delay,
 *   so a burst of reveals triggers a single rescan shortly after the last
 *   one.
 *
 * `effect` may return a cleanup function (e.g. to disconnect an
 * IntersectionObserver); it is invoked before the next run and on unmount.
 */
function useCoalescedRevealEffect(
    effect: () => void | (() => void),
    syncKey: unknown,
    renderedSectionCount: number,
) {
    const effectRef = useRef(effect);
    useEffect(() => {
        effectRef.current = effect;
    });

    const cleanupRef = useRef<(() => void) | null>(null);
    const previousSyncKeyRef = useRef<unknown>(NEVER_RAN);

    const runEffect = useCallback(() => {
        cleanupRef.current?.();
        const cleanup = effectRef.current();
        cleanupRef.current = typeof cleanup === "function" ? cleanup : null;
    }, []);

    useEffect(() => {
        // `renderedSectionCount` is intentionally an effect trigger rather
        // than an input: each progressive reveal restarts the trailing
        // debounce below.
        void renderedSectionCount;

        if (!Object.is(previousSyncKeyRef.current, syncKey)) {
            previousSyncKeyRef.current = syncKey;
            runEffect();
            return;
        }

        const timeoutId = window.setTimeout(runEffect, REVEAL_RESCAN_DELAY_MS);
        return () => window.clearTimeout(timeoutId);
    }, [runEffect, syncKey, renderedSectionCount]);

    // Run the last effect cleanup (e.g. disconnect the observer) on unmount.
    useEffect(
        () => () => {
            cleanupRef.current?.();
            cleanupRef.current = null;
        },
        [],
    );
}

const Notie: React.FC<NotieProps> = ({
    markdown,
    config: userConfig,
    theme = "default",
    customComponents: userCustomComponents,
}) => {
    const config = useNotieConfig(userConfig, theme);
    // `customComponents` values are functions, so we cannot stabilize them
    // structurally (e.g. via JSON). Instead, reuse the previous object
    // identity when the object stays shallowly equal (same keys, same
    // function references), so inline `customComponents={{...}}` literals do
    // not invalidate memoized children on every parent render. Consumers who
    // pass freshly created functions each render still opt out of memoization.
    const customComponents = useShallowStableObject(userCustomComponents);
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

        const scrollToTarget = (target: Element) => {
            target.scrollIntoView();
            if (window.location.hash !== `#${pendingScrollId}`) {
                window.history.pushState(null, "", `#${pendingScrollId}`);
            }
            setPendingScrollId(null);
        };

        const target = document.getElementById(pendingScrollId);
        if (target) {
            scrollToTarget(target);
            return;
        }

        // The target may not exist yet even after every section has been
        // revealed: anchor ids such as `eqn-X.Y` are assigned by the
        // coalesced DOM-labeling effects above, which trail the final
        // `renderedSectionCount` update by a short debounce. Since this
        // effect gets no further dependency changes at that point, poll
        // briefly for the anchor instead of giving up, bounded so an id
        // that never materializes cannot poll forever.
        const RETRY_INTERVAL_MS = 80;
        const RETRY_TIMEOUT_MS = 2000;
        const deadline = Date.now() + RETRY_TIMEOUT_MS;
        let timeoutId: number | undefined;
        const retry = () => {
            const retryTarget = document.getElementById(pendingScrollId);
            if (retryTarget) {
                scrollToTarget(retryTarget);
                return;
            }
            if (Date.now() >= deadline) return;
            timeoutId = window.setTimeout(retry, RETRY_INTERVAL_MS);
        };
        timeoutId = window.setTimeout(retry, RETRY_INTERVAL_MS);
        return () => window.clearTimeout(timeoutId);
    }, [pendingScrollId, renderedSectionCount]);

    // Effect to observe headings and update activeId. Coalesced so that a
    // burst of progressive section reveals rebuilds the observer once at the
    // end instead of once per revealed section.
    useCoalescedRevealEffect(
        useCallback(() => {
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
                        setActiveId((current) =>
                            current === id ? current : id,
                        );
                    }
                });
            }, observerOptions);

            headings.forEach((heading) => observer.observe(heading));

            return () => observer.disconnect();
        }, []),
        markdownContent,
        renderedSectionCount,
    );

    // Effect to auto label equation numbers. Coalesced: see above.
    const labelEquationNumbers = useCallback(() => {
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
    }, []);
    useCoalescedRevealEffect(
        labelEquationNumbers,
        markdownContent,
        renderedSectionCount,
    );

    // Effect to auto label Definitions, Theorems, Lemmas, only for LaTeX
    // style. Coalesced: see above.
    const labelLatexBlockquotes = useCallback(() => {
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
    }, [config.theme.blockquoteStyle]);
    useCoalescedRevealEffect(
        labelLatexBlockquotes,
        // Re-run synchronously when either the document content or the
        // blockquote style changes (string sync keys compare by value).
        `${config.theme.blockquoteStyle}\n${markdownContent}`,
        renderedSectionCount,
    );

    // Skip the TOC column entirely when there is nothing to list (empty
    // markdown or title-only documents), so a bare "Contents" heading never
    // renders next to the note.
    const showToc = config.showTableOfContents && tocEntries.length > 0;

    return (
        <Pane className={styles["notie-container"]}>
            <Pane className={showToc ? styles["mw-page-container-inner"] : ""}>
                {showToc && (
                    <Pane className={styles["vector-column-start"]}>
                        <NoteToc
                            tocEntries={tocEntries}
                            activeId={activeId}
                            tocTitle={config.tocTitle}
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
