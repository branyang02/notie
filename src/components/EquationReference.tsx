import { Card, Tooltip } from "evergreen-ui";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { extractEquationInfo, processEquationString } from "../utils";

const EquationReference: React.FC<{
    children: Element;
    equationMapping: {
        [key: string]: {
            equationNumber: string;
            equationString: string;
        };
    };
    previewEquation?: boolean;
}> = ({ children, equationMapping, previewEquation }) => {
    const { equationNumber, equationString, parenthesesRemoved } =
        extractEquationInfo(children, equationMapping);

    if (equationString == "error") {
        return (
            <span className="mord" style={{ color: "red" }}>
                {parenthesesRemoved ? `(${equationNumber})` : equationNumber}
            </span>
        );
    }

    return previewEquation ? (
        <Tooltip
            content={<EquationCard equationString={equationString} />}
            appearance="card"
            statelessProps={{
                maxWidth: "100%",
            }}
        >
            <a href={`#eqn-${equationNumber}`}>
                <span className="mord">
                    {parenthesesRemoved
                        ? `(${equationNumber})`
                        : equationNumber}
                </span>
            </a>
        </Tooltip>
    ) : (
        <a href={`#eqn-${equationNumber}`}>
            <span className="mord">
                {parenthesesRemoved ? `(${equationNumber})` : equationNumber}
            </span>
        </a>
    );
};

export default EquationReference;

const EquationCard = ({ equationString }: { equationString: string }) => {
    const processedEquationString = processEquationString(equationString);

    return (
        <Card>
            <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[
                    [rehypeKatex],
                    rehypeRaw,
                    rehypeHighlight,
                    rehypeSlug,
                ]}
            >
                {processedEquationString}
            </ReactMarkdown>
        </Card>
    );
};
