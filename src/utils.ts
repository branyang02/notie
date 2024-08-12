export interface EquationMapping {
    [key: string]: {
        equationNumber: string;
        equationString: string;
    };
}

// Used in EquationReference.tsx
export function extractEquationInfo(
    children: Element,
    equationMapping: EquationMapping,
) {
    const equationLabel = children.textContent?.replace(/−/g, "-") || "";
    if (!equationLabel) {
        throw new Error("No equation label found");
    }

    const trimmedLabel = equationLabel.replace(/^\(|\)$/g, "");
    const parenthesesRemoved = trimmedLabel !== equationLabel;

    if (!(trimmedLabel in equationMapping)) {
        console.error(
            `Equation label "${trimmedLabel}" not found in equation mapping`,
        );

        return {
            equationNumber: `Error: reference ${trimmedLabel} not labeled`,
            equationString: "error",
            parenthesesRemoved: false,
        };
    }

    const { equationNumber, equationString } = equationMapping[trimmedLabel];

    return { equationNumber, equationString, parenthesesRemoved };
}

// Used in EquationReference.tsx
export function processEquationString(equationString: string): string {
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
            .replace(/&=/g, "=");
        processedEquationString += "\n$$\n";
    }
    return processedEquationString;
}
