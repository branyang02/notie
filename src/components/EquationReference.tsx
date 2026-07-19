import { Card, Tooltip } from "evergreen-ui";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { katexOptions } from "../utils/katexOptions";
import { sanitizeUrl } from "../utils/sanitizeUrl";
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
                {/* Screen-reader prefix so the error state is not conveyed
                    by the red color alone. */}
                <span className="sr-only">Unresolved reference: </span>
                {parenthesesRemoved ? `(${equationNumber})` : equationNumber}
            </span>
        );
    }

    const ariaLabel = `Equation ${equationNumber}`;

    return previewEquation ? (
        <Tooltip
            content={<EquationCard equationString={equationString} />}
            appearance="card"
            statelessProps={{
                maxWidth: "100%",
            }}
        >
            <a href={`#eqn-${equationNumber}`} aria-label={ariaLabel}>
                <span className="mord">
                    {parenthesesRemoved
                        ? `(${equationNumber})`
                        : equationNumber}
                </span>
            </a>
        </Tooltip>
    ) : (
        <a href={`#eqn-${equationNumber}`} aria-label={ariaLabel}>
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
                urlTransform={sanitizeUrl}
            >
                {processedEquationString}
            </ReactMarkdown>
        </Card>
    );
};
