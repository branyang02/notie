export interface BlockquoteMapping {
    [label: string]: {
        blockquoteNumber: string;
        blockquoteType: string;
        blockquoteContent: string;
    };
}

export function extractBlockquoteInfo(
    href: string | null | undefined,
    blockquoteMapping: BlockquoteMapping,
) {
    const label = href?.split("#bqref-").pop();
    if (!label) throw new Error("No blockquote label found");

    if (!(label in blockquoteMapping)) {
        console.error(
            `Blockquote label "${label}" not found in blockquote mapping`,
        );
        return {
            blockquoteNumber: `Error: reference ${label} not labeled`,
            blockquoteType: "unknown",
            blockquoteContent: "error",
        };
    }

    const { blockquoteNumber, blockquoteType, blockquoteContent } =
        blockquoteMapping[label];
    return { blockquoteNumber, blockquoteType, blockquoteContent };
}

export interface EquationMapping {
    [key: string]: {
        equationNumber: string;
        equationString: string;
    };
}

function parseEquationReferenceHref(href: string | null | undefined) {
    const rawLabel = href?.split("#pre-eqn-").pop();
    if (!rawLabel) return null;

    if (rawLabel.startsWith("eqref:")) {
        return {
            label: rawLabel.slice("eqref:".length),
            parenthesesRemoved: true,
        };
    }

    if (rawLabel.startsWith("ref:")) {
        return {
            label: rawLabel.slice("ref:".length),
            parenthesesRemoved: false,
        };
    }

    return {
        label: rawLabel,
        parenthesesRemoved: undefined,
    };
}

// Used in EquationReference.tsx
export function extractEquationInfo(
    href: string | null | undefined,
    textContent: string | null | undefined,
    equationMapping: EquationMapping,
) {
    const parsedReference = parseEquationReferenceHref(href);
    if (!parsedReference) {
        throw new Error("No equation label found");
    }
    const { label } = parsedReference;

    const parenthesesRemoved =
        parsedReference.parenthesesRemoved ??
        textContent?.includes("(") ??
        false;

    if (!(label in equationMapping)) {
        console.error(
            `Equation label "${label}" not found in equation mapping`,
        );

        return {
            equationNumber: `Error: reference ${label} not labeled`,
            equationString: "error",
            parenthesesRemoved,
        };
    }

    const { equationNumber, equationString } = equationMapping[label];

    return { equationNumber, equationString, parenthesesRemoved };
}

// Used in EquationReference.tsx
export function processEquationString(equationString: string): string {
    // equationString is a line of LaTeX equation.
    let processedEquationString = "";
    // Use `aligned` environment even though we only have one line of equation. This is because we do not have to deal with `&` symbols in the equation.
    processedEquationString += "$$\n\\begin{aligned}\n";
    processedEquationString += equationString.replace(/\\label\{[^}]*\}/g, ""); // Remove \label{...}
    processedEquationString += "\n\\end{aligned}\n$$\n";

    return processedEquationString;
}
