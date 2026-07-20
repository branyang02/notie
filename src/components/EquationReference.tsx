import { Card, Tooltip } from "evergreen-ui";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { katexOptions } from "../utils/katexOptions";
import rehypeAccessibleKatexRefs from "../utils/rehypeAccessibleKatexRefs";
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
    inert,
}: {
    href: string;
    textContent?: string;
    equationMapping: EquationMapping;
    previewEquation?: boolean;
    /**
     * True when this reference stays inside KaTeX's `aria-hidden` HTML
     * subtree (a `\eqref`/`\ref` embedded in a larger formula, flagged by
     * the rehypeAccessibleKatexRefs plugin). The anchor is then removed
     * from the tab order so no focusable element hides inside an
     * `aria-hidden` container; it remains clickable with a pointer.
     */
    inert?: boolean;
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
    // Inside an aria-hidden KaTeX subtree the anchor must not be reachable
    // via keyboard (WCAG: no focusable content hidden from AT); it is
    // already hidden from the accessibility tree by the ancestor.
    const tabIndex = inert ? -1 : undefined;

    return previewEquation ? (
        <Tooltip
            content={<EquationCard equationString={equationString} />}
            appearance="card"
            statelessProps={{
                maxWidth: "100%",
            }}
        >
            <a
                href={`#eqn-${equationNumber}`}
                aria-label={ariaLabel}
                tabIndex={tabIndex}
            >
                <span className="mord">
                    {parenthesesRemoved
                        ? `(${equationNumber})`
                        : equationNumber}
                </span>
            </a>
        </Tooltip>
    ) : (
        <a
            href={`#eqn-${equationNumber}`}
            aria-label={ariaLabel}
            tabIndex={tabIndex}
        >
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
                    [rehypeKatex, katexOptions],
                    rehypeAccessibleKatexRefs,
                ]}
                urlTransform={sanitizeUrl}
            >
                {processedEquationString}
            </ReactMarkdown>
        </Card>
    );
};
