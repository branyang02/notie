import { useEffect, useRef, useState } from "react";
import { Pane } from "evergreen-ui";
import styles from "../styles/Notie.module.css";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const Desmos: any;

/**
 * Desmos's public demo API key. Calculators loaded with it log a warning
 * that it is not licensed for commercial use; consumers can supply their
 * own key via `NotieConfig.desmosApiKey`.
 */
export const DEFAULT_DESMOS_API_KEY = "dcb31709b452b1cf9dc26972add0fda6";

// One cached loader promise per API key: different keys use different
// script URLs, so they must not share a cache entry.
const desmosScriptPromises = new Map<string, Promise<void>>();

function loadDesmosScript(apiKey: string): Promise<void> {
    if (typeof Desmos !== "undefined") return Promise.resolve();

    let scriptPromise = desmosScriptPromises.get(apiKey);
    if (!scriptPromise) {
        scriptPromise = new Promise<void>((resolve, reject) => {
            const scriptEl = document.createElement("script");
            scriptEl.src = `https://www.desmos.com/api/v1.9/calculator.js?apiKey=${encodeURIComponent(apiKey)}`;
            scriptEl.async = true;
            scriptEl.onload = () => resolve();
            scriptEl.onerror = () =>
                reject(new Error("Failed to load Desmos script"));
            document.head.appendChild(scriptEl);
        }).catch((error) => {
            // Do not cache the rejection, so the next mount retries.
            desmosScriptPromises.delete(apiKey);
            throw error;
        });
        desmosScriptPromises.set(apiKey, scriptPromise);
    }

    return scriptPromise;
}

export interface DesmosGraphProps {
    graphScript: string;
    appearance?: "light" | "dark";
    /**
     * Desmos calculator API key. Defaults to the built-in demo key, which
     * logs a warning that it is not licensed for commercial use.
     */
    apiKey?: string;
}

const DesmosGraph = ({
    graphScript,
    appearance = "light",
    apiKey = DEFAULT_DESMOS_API_KEY,
}: DesmosGraphProps) => {
    const calculatorRef = useRef<HTMLDivElement | null>(null);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const calculatorInstance = useRef<any>(null); // Store the Desmos instance
    const [loadFailed, setLoadFailed] = useState(false);

    useEffect(() => {
        let cancelled = false;

        setLoadFailed(false);

        loadDesmosScript(apiKey)
            .then(() => {
                if (cancelled) return;
                if (calculatorRef.current && !calculatorInstance.current) {
                    calculatorInstance.current = Desmos.GraphingCalculator(
                        calculatorRef.current,
                        // Match the surrounding theme in dark mode.
                        { invertedColors: appearance === "dark" },
                    );
                    updateExpressions();
                }
            })
            .catch((error) => {
                console.error(error);
                if (!cancelled) setLoadFailed(true);
            });

        return () => {
            cancelled = true;
            calculatorInstance.current?.destroy?.();
            calculatorInstance.current = null;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [appearance, apiKey]);

    // Update the calculator when graphScript changes
    useEffect(() => {
        if (calculatorInstance.current) {
            updateExpressions();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [graphScript]);

    const updateExpressions = () => {
        const equationList = graphScript
            .split("\n")
            .filter((line) => line.trim());
        equationList.forEach((equation, index) => {
            calculatorInstance.current.setExpression({
                id: `graph${index}`,
                latex: equation,
            });
        });
    };

    if (loadFailed) {
        return (
            <Pane
                maxWidth="100%"
                className={styles["diagram-fallback"]}
                data-testid="desmos-fallback"
            >
                <div>Failed to load Desmos.</div>
            </Pane>
        );
    }

    return (
        <Pane
            ref={calculatorRef}
            display="flex"
            justifyContent="center"
            alignItems="center"
            flexGrow={1}
            maxWidth="100%"
            height="400px"
            className={styles["desmos-graph"]}
        ></Pane>
    );
};

export default DesmosGraph;
