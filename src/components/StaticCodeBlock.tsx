import { Pane } from "evergreen-ui";
import { atomOneLight, CodeBlock, CopyBlock, nord } from "react-code-blocks";

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
      <Pane
        className="language-box"
        paddingY={1}
        paddingX={8}
        style={{
          fontSize: "0.8rem",
          borderRadius: "10px 10px 0 0",
          backgroundColor: "#afb8c133",
        }}
      >
        {""}
        {language}
      </Pane>
      <div className="code-blocks">
        {code.includes("\n") ? (
          <CopyBlock
            text={code}
            language={language}
            showLineNumbers={false}
            theme={darkMode ? nord : atomOneLight}
            startingLineNumber={0}
            customStyle={{ borderRadius: "0 0 10px 10px" }}
          />
        ) : (
          <CodeBlock
            text={code}
            language={language}
            showLineNumbers={false}
            theme={darkMode ? nord : atomOneLight}
            startingLineNumber={0}
            customStyle={{ borderRadius: "0 0 10px 10px" }}
          />
        )}
      </div>
    </Pane>
  );
};

export default StaticCodeBlock;
