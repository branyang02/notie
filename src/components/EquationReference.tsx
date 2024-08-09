import { Tooltip } from "evergreen-ui";
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
}> = ({ children, equationMapping }) => {
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

    return (
        <Tooltip
            // content={<Paragraph margin={40}>Card appearance</Paragraph>}
            content={<EquationCard equationString={equationString} />}
            appearance="card"
        >
            <a href={`#eqn-${equationNumber}`}>
                <span className="mord">
                    {parenthesesRemoved
                        ? `(${equationNumber})`
                        : equationNumber}
                </span>
            </a>
        </Tooltip>
    );
};

export default EquationReference;

const EquationCard = ({ equationString }: { equationString: string }) => {
    // Process `equationString`
    const processedEquationString = equationString
        .replace(/\\label\{[^}]*\}/g, "")
        .replace(/\\begin\{align\}/g, "\\begin{aligned}")
        .replace(/\\begin\{equation\}/g, "")
        .replace(/\\end\{equation\}/g, "");

    console.log(processedEquationString);

    return (
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
    );
};
