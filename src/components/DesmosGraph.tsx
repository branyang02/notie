import { useEffect, useRef } from "react";
import { Pane } from "evergreen-ui";
import styles from "../styles/Notie.module.css";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const Desmos: any;

let desmosScriptPromise: Promise<void> | null = null;

function loadDesmosScript(): Promise<void> {
    if (typeof Desmos !== "undefined") return Promise.resolve();

    if (!desmosScriptPromise) {
        desmosScriptPromise = new Promise((resolve, reject) => {
            const scriptEl = document.createElement("script");
            scriptEl.src =
                "https://www.desmos.com/api/v1.9/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6";
            scriptEl.async = true;
            scriptEl.onload = () => resolve();
            scriptEl.onerror = () =>
                reject(new Error("Failed to load Desmos script"));
            document.head.appendChild(scriptEl);
        });
    }

    return desmosScriptPromise;
}

const DesmosGraph = ({ graphScript }: { graphScript: string }) => {
    const calculatorRef = useRef<HTMLDivElement | null>(null);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const calculatorInstance = useRef<any>(null); // Store the Desmos instance

    useEffect(() => {
        let cancelled = false;

        loadDesmosScript()
            .then(() => {
                if (cancelled) return;
                if (calculatorRef.current && !calculatorInstance.current) {
                    calculatorInstance.current = Desmos.GraphingCalculator(
                        calculatorRef.current,
                    );
                    updateExpressions();
                }
            })
            .catch((error) => console.error(error));

        return () => {
            cancelled = true;
            calculatorInstance.current?.destroy?.();
            calculatorInstance.current = null;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

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
