import { Card, Tooltip } from "evergreen-ui";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

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
    const equationLabel = children.textContent?.replace(/−/g, "-") || "";
    if (!equationLabel) {
        return null;
    }

    const trimmedLabel = equationLabel.replace(/^\(|\)$/g, "");
    const parenthesesRemoved = trimmedLabel !== equationLabel;

    if (!(trimmedLabel in equationMapping)) {
        throw new Error(
            `Equation label "${trimmedLabel}" not found in equation mapping`,
        );
    }

    const { equationNumber, equationString } = equationMapping[trimmedLabel];

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
    // Process `equationString`

    console.log(equationString);

    let processedEquationString = "";
    if (equationString.includes("\\begin{equation}")) {
        processedEquationString = equationString
            .replace(/\\label\{[^}]*\}/g, "")
            .replace(/\\begin\{align\}/g, "\\begin{aligned}")
            .replace(/\\begin\{equation\}/g, "")
            .replace(/\\end\{equation\}/g, "");
    } else {
        // We are given a single line from \begin{align}
        processedEquationString += "$$\n";

        processedEquationString += equationString
            .replace(/\\label\{[^}]*\}/g, "")
            .replace(/&/g, "");

        processedEquationString += "\n$$\n";
    }

    console.log(processedEquationString);

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
