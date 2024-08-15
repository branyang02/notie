import { Pane } from "evergreen-ui";
import CodeHeader from "./CodeHeader";
import SyntaxHighlighter from "react-syntax-highlighter";
import * as themes from "react-syntax-highlighter/dist/esm/styles/hljs";

const StaticCodeBlock = ({
    code,
    language,
    theme,
    copyButtonHoverColor,
}: {
    code: string;
    language: string;
    theme: string;
    copyButtonHoverColor?: string;
}) => {
    let selectedTheme = themes[theme as keyof typeof themes];
    if (!selectedTheme) {
        console.error(`Invalid theme name: ${theme}, falling back to default.`);
        selectedTheme = themes.github; // Default fallback theme
    }

    return (
        <Pane>
            <CodeHeader
                language={language}
                code={code}
                copyButtonHoverColor={copyButtonHoverColor}
            />
            <div className="code-blocks">
                <SyntaxHighlighter
                    language="language"
                    style={selectedTheme}
                    customStyle={{
                        fontSize: "1em",
                        borderRadius: "0 0 10px 10px",
                        marginTop: 0,
                    }}
                >
                    {code}
                </SyntaxHighlighter>
            </div>
        </Pane>
    );
};

export default StaticCodeBlock;
