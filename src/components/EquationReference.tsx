import { Card, Tooltip } from "evergreen-ui";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { katexOptions } from "../utils/katexOptions";
import {
    EquationMapping,
    extractEquationInfo,
    processEquationString,
} from "../utils/utils";

const EquationReference = ({
    href,
    textContent,
    equationMapping,
    previewEquation,
}: {
    href: string;
    textContent?: string;
    equationMapping: EquationMapping;
    previewEquation?: boolean;
}) => {
    const { equationNumber, equationString, parenthesesRemoved } =
        extractEquationInfo(href, textContent, equationMapping);

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
                rehypePlugins={[[rehypeKatex, katexOptions]]}
            >
                {processedEquationString}
            </ReactMarkdown>
        </Card>
    );
};
