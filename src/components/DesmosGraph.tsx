import { useEffect, useRef } from "react";
import { Pane } from "evergreen-ui";
import styles from "../styles/Notie.module.css";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const Desmos: any;

const DesmosGraph = ({ graphScript }: { graphScript: string }) => {
    const calculatorRef = useRef<HTMLDivElement | null>(null);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const calculatorInstance = useRef<any>(null); // Store the Desmos instance
    const scriptLoaded = useRef(false); // Track if the script is loaded

    useEffect(() => {
        if (!scriptLoaded.current) {
            const scriptEl = document.createElement("script");
            const apiUrl = import.meta.env.VITE_DESMOS_API_URL;
            scriptEl.src = apiUrl;
            scriptEl.async = true;
            scriptEl.onload = () => {
                scriptLoaded.current = true;
                if (calculatorRef.current && !calculatorInstance.current) {
                    calculatorInstance.current = Desmos.GraphingCalculator(
                        calculatorRef.current,
                    );
                }
                updateExpressions(); // Initial update
            };
            document.head.appendChild(scriptEl);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Empty dependency array ensures this only runs once

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
