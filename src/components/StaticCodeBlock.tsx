import { Pane } from "evergreen-ui";
import * as ReactCodeBlocks from "react-code-blocks";
import CodeHeader from "./CodeHeader";

const StaticCodeBlock = ({
    code,
    language,
    theme,
}: {
    code: string;
    language: string;
    theme: string;
}) => {
    let selectedTheme;

    try {
        selectedTheme = ReactCodeBlocks[theme as keyof typeof ReactCodeBlocks];
    } catch (error) {
        console.error(`Invalid theme name: ${theme}, falling back to default.`);
        selectedTheme = ReactCodeBlocks.atomOneLight; // Default fallback theme
    }
    return (
        <Pane>
            <CodeHeader language={language} code={code} />
            <div className="code-blocks">
                <ReactCodeBlocks.CodeBlock
                    text={code}
                    language={language}
                    showLineNumbers={false}
                    theme={selectedTheme}
                    startingLineNumber={0}
                    customStyle={{ borderRadius: "0 0 10px 10px" }}
                />
            </div>
        </Pane>
    );
};

export default StaticCodeBlock;
