import { Pane } from "evergreen-ui";
import { useEffect, useRef } from "react";

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

const TikZ = ({ tikzScript }: { tikzScript: string }) => {
    const scriptContainerRef = useRef<HTMLDivElement>(null);
    const scriptLoaded = useRef(false);
    const cssLoaded = useRef(false);

    useEffect(() => {
        // Load CSS if not already loaded
        if (!cssLoaded.current) {
            fetch(
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
                    cssLoaded.current = true;
                })
                .catch((error) => console.error("Failed to load CSS", error));
        }

        // Load script if not already loaded
        if (!scriptLoaded.current) {
            fetch(
                "https://raw.githubusercontent.com/artisticat1/obsidian-tikzjax/main/tikzjax.js",
            )
                .then((response) => response.text())
                .then((scriptContent) => {
                    const scriptEl = document.createElement("script");
                    scriptEl.textContent = scriptContent;
                    document.body.appendChild(scriptEl);
                    scriptLoaded.current = true;
                })
                .catch((error) =>
                    console.error("Failed to load script", error),
                );
        }

        // Cleanup on component unmount
        return () => {
            if (scriptContainerRef.current) {
                // Remove any existing TikZ scripts when component unmounts
                const existingScripts =
                    scriptContainerRef.current.querySelectorAll(
                        'script[type="text/tikz"]',
                    );
                existingScripts.forEach((script) => script.remove());
            }
        };
    }, []);

    useEffect(() => {
        function loadTikZJax() {
            const scriptEl = document.createElement("script");
            scriptEl.type = "text/tikz";
            scriptEl.async = true;
            scriptEl.textContent = tidyTikzSource(tikzScript);

            scriptEl.setAttribute("data-show-console", "true");

            if (scriptContainerRef.current) {
                // Remove any previous TikZ scripts before adding a new one
                const existingScripts =
                    scriptContainerRef.current.querySelectorAll(
                        'script[type="text/tikz"]',
                    );
                existingScripts.forEach((script) => script.remove());

                scriptContainerRef.current.appendChild(scriptEl);
            }
        }

        loadTikZJax();
    }, [tikzScript]);

    return (
        <Pane
            ref={scriptContainerRef}
            className="tikz-drawing"
            display="flex"
            justifyContent="center"
            alignItems="center"
            flexGrow={1}
        ></Pane>
    );
};

export default TikZ;
