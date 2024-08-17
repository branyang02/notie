import { Pane } from "evergreen-ui";
import CodeHeader from "./CodeHeader";
import SyntaxHighlighter from "react-syntax-highlighter";
import * as themes from "react-syntax-highlighter/dist/esm/styles/hljs";
import styles from "../styles/Notie.module.css";

const StaticCodeBlock = ({
    code,
    language,
    theme,
}: {
    code: string;
    language: string;
    theme: string;
}) => {
    let selectedTheme = themes[theme as keyof typeof themes];
    if (!selectedTheme) {
        console.error(`Invalid theme name: ${theme}, falling back to default.`);
        selectedTheme = themes.github; // Default fallback theme
    }

    return (
        <Pane>
            <CodeHeader language={language} code={code} />
            <div className={styles["code-blocks"]}>
                <SyntaxHighlighter
                    language="language"
                    style={selectedTheme}
                    customStyle={{
                        borderRadius: "0 0 10px 10px",
                        marginTop: 0,
                        marginBottom: 0,
                    }}
                >
                    {code}
                </SyntaxHighlighter>
            </div>
        </Pane>
    );
};

export default StaticCodeBlock;
