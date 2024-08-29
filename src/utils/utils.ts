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
    const href = children.getAttribute("href");
    const label = href?.split("#pre-eqn-").pop();
    if (!label) {
        throw new Error("No equation label found");
    }

    const parenthesesRemoved = children.textContent?.includes("(") ?? false;

    if (!(label in equationMapping)) {
        console.error(
            `Equation label "${label}" not found in equation mapping`,
        );

        return {
            equationNumber: `Error: reference ${label} not labeled`,
            equationString: "error",
            parenthesesRemoved: false,
        };
    }

    const { equationNumber, equationString } = equationMapping[label];

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
