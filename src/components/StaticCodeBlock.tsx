import { useEffect, useState } from "react";
import { Pane } from "evergreen-ui";
import CodeHeader from "./CodeHeader";
import styles from "../styles/Notie.module.css";
import { highlightWithCache } from "../utils/shikiHighlighter";

function escapeHtml(s: string) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

const StaticCodeBlock = ({
    code,
    language,
    theme,
}: {
    code: string;
    language: string;
    theme: string;
}) => {
    const [html, setHtml] = useState<string>(
        `<pre><code>${escapeHtml(code)}</code></pre>`,
    );

    useEffect(() => {
        let cancelled = false;
        highlightWithCache(code, language, theme)
            .then((highlighted) => {
                if (cancelled) return;
                setHtml(highlighted);
            })
            .catch(() => {
                // keep escaped fallback
            });
        return () => {
            cancelled = true;
        };
    }, [code, language, theme]);

    return (
        <Pane>
            <CodeHeader language={language} code={code} />
            <div
                className={styles["code-blocks"]}
                dangerouslySetInnerHTML={{ __html: html }}
            />
        </Pane>
    );
};

export default StaticCodeBlock;
