import { Pane } from "evergreen-ui";
import { atomOneLight, CodeBlock, nord } from "react-code-blocks";
import CodeHeader from "./CodeHeader";

const StaticCodeBlock = ({
    code,
    language,
    darkMode = false,
}: {
    code: string;
    language: string;
    darkMode?: boolean;
}) => {
    return (
        <Pane>
            <CodeHeader language={language} code={code} darkMode={darkMode} />
            <div className="code-blocks">
                <CodeBlock
                    text={code}
                    language={language}
                    showLineNumbers={false}
                    theme={darkMode ? nord : atomOneLight}
                    startingLineNumber={0}
                    customStyle={{ borderRadius: "0 0 10px 10px" }}
                />
            </div>
        </Pane>
    );
};

export default StaticCodeBlock;
