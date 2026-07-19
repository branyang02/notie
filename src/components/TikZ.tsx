import { Pane } from "evergreen-ui";
import { useEffect, useRef, useState } from "react";
import styles from "../styles/Notie.module.css";

// TikZJax assets pinned to obsidian-tikzjax commit
// 1d1f0844bd918e09e7eac081f86a70ba28635301 (main as of 2024-07-13) so that
// upstream changes cannot break rendering.
const TIKZJAX_CSS_URL =
    "https://raw.githubusercontent.com/artisticat1/obsidian-tikzjax/1d1f0844bd918e09e7eac081f86a70ba28635301/styles.css";
const TIKZJAX_SCRIPT_URL =
    "https://raw.githubusercontent.com/artisticat1/obsidian-tikzjax/1d1f0844bd918e09e7eac081f86a70ba28635301/tikzjax.js";

let tikzCssPromise: Promise<void> | null = null;
let tikzScriptPromise: Promise<void> | null = null;

function loadTikzCss(): Promise<void> {
    if (!tikzCssPromise) {
        tikzCssPromise = fetch(TIKZJAX_CSS_URL)
            .then((response) => {
                if (response.ok) return response.text();
                throw new Error("Failed to load CSS");
            })
            .then((cssContent) => {
                const styleEl = document.createElement("style");
                styleEl.textContent = cssContent;
                document.head.appendChild(styleEl);
            })
            .catch((error) => {
                // Do not cache the rejection, so the next mount retries.
                tikzCssPromise = null;
                throw error;
            });
    }

    return tikzCssPromise;
}

function loadTikzScript(): Promise<void> {
    if (!tikzScriptPromise) {
        tikzScriptPromise = fetch(TIKZJAX_SCRIPT_URL)
            .then((response) => {
                if (response.ok) return response.text();
                throw new Error("Failed to load script");
            })
            .then((scriptContent) => {
                const scriptEl = document.createElement("script");
                scriptEl.textContent = scriptContent;
                document.body.appendChild(scriptEl);
            })
            .catch((error) => {
                // Do not cache the rejection, so the next mount retries.
                tikzScriptPromise = null;
                throw error;
            });
    }

    return tikzScriptPromise;
}

function tidyTikzSource(tikzSource: string) {
    // Remove non-breaking space characters, otherwise we get errors
    const remove = /&nbsp;/g;
    tikzSource = tikzSource.replace(remove, "");

    let lines = tikzSource.split("\n");

    // Trim whitespace that is inserted when pasting in code, otherwise TikZJax complains
    lines = lines.map((line) => line.trim());

    // Remove empty lines
    lines = lines.filter((line) => line);

    return lines.join("\n");
}

function clearContainer(container: HTMLDivElement | null) {
    if (!container) return;

    // Remove both pending text/tikz scripts and previously rendered SVGs so
    // diagrams do not stack up when the source changes.
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }
}

function waitForNextFrame(): Promise<void> {
    if (typeof window === "undefined") return Promise.resolve();

    return new Promise((resolve) => {
        window.requestAnimationFrame(() => resolve());
    });
}

const TikZ = ({ tikzScript }: { tikzScript: string }) => {
    const scriptContainerRef = useRef<HTMLDivElement>(null);
    const [loadFailed, setLoadFailed] = useState(false);

    useEffect(() => {
        const container = scriptContainerRef.current;
        let isCancelled = false;

        setLoadFailed(false);

        async function renderTikz() {
            try {
                await Promise.all([loadTikzCss(), loadTikzScript()]);
                await waitForNextFrame();
            } catch (error) {
                console.error("Failed to load TikZJax", error);
                if (!isCancelled) setLoadFailed(true);
                return;
            }

            if (isCancelled || !container) return;

            const scriptEl = document.createElement("script");
            scriptEl.type = "text/tikz";
            scriptEl.async = true;
            scriptEl.textContent = tidyTikzSource(tikzScript);
            scriptEl.setAttribute("data-show-console", "true");

            // TikZJax registers a MutationObserver after its loader runs, so
            // append only after the loader promise resolves.
            clearContainer(container);
            container.appendChild(scriptEl);
        }

        renderTikz();

        return () => {
            isCancelled = true;
            clearContainer(container);
        };
    }, [tikzScript]);

    return (
        <>
            {loadFailed && (
                <Pane
                    maxWidth="100%"
                    className={styles["diagram-fallback"]}
                    data-testid="tikz-fallback"
                >
                    <div>TikZ rendering failed. Showing source instead.</div>
                    <pre>{tikzScript}</pre>
                </Pane>
            )}
            <Pane
                ref={scriptContainerRef}
                display={loadFailed ? "none" : "flex"}
                justifyContent="center"
                alignItems="center"
                flexGrow={1}
                maxWidth="100%"
                className={styles["tikz-drawing"]}
            ></Pane>
        </>
    );
};

export default TikZ;
