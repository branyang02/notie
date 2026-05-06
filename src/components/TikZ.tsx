import { Pane } from "evergreen-ui";
import { useEffect, useRef } from "react";
import styles from "../styles/Notie.module.css";

let tikzCssPromise: Promise<void> | null = null;
let tikzScriptPromise: Promise<void> | null = null;

function loadTikzCss(): Promise<void> {
    if (!tikzCssPromise) {
        tikzCssPromise = fetch(
            "https://raw.githubusercontent.com/artisticat1/obsidian-tikzjax/main/styles.css",
        )
            .then((response) => {
                if (response.ok) return response.text();
                throw new Error("Failed to load CSS");
            })
            .then((cssContent) => {
                const styleEl = document.createElement("style");
                styleEl.textContent = cssContent;
                document.head.appendChild(styleEl);
            });
    }

    return tikzCssPromise;
}

function loadTikzScript(): Promise<void> {
    if (!tikzScriptPromise) {
        tikzScriptPromise = fetch(
            "https://raw.githubusercontent.com/artisticat1/obsidian-tikzjax/main/tikzjax.js",
        )
            .then((response) => {
                if (response.ok) return response.text();
                throw new Error("Failed to load script");
            })
            .then((scriptContent) => {
                const scriptEl = document.createElement("script");
                scriptEl.textContent = scriptContent;
                document.body.appendChild(scriptEl);
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

function removeTikzScripts(container: HTMLDivElement | null) {
    if (!container) return;

    const existingScripts = container.querySelectorAll(
        'script[type="text/tikz"]',
    );
    existingScripts.forEach((script) => script.remove());
}

function waitForNextFrame(): Promise<void> {
    if (typeof window === "undefined") return Promise.resolve();

    return new Promise((resolve) => {
        window.requestAnimationFrame(() => resolve());
    });
}

const TikZ = ({ tikzScript }: { tikzScript: string }) => {
    const scriptContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const container = scriptContainerRef.current;
        let isCancelled = false;

        async function renderTikz() {
            try {
                await Promise.all([loadTikzCss(), loadTikzScript()]);
                await waitForNextFrame();
            } catch (error) {
                console.error("Failed to load TikZJax", error);
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
            removeTikzScripts(container);
            container.appendChild(scriptEl);
        }

        renderTikz();

        return () => {
            isCancelled = true;
            removeTikzScripts(container);
        };
    }, [tikzScript]);

    return (
        <Pane
            ref={scriptContainerRef}
            display="flex"
            justifyContent="center"
            alignItems="center"
            flexGrow={1}
            maxWidth="100%"
            className={styles["tikz-drawing"]}
        ></Pane>
    );
};

export default TikZ;
