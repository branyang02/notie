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
    // List of addition KaTeX environments where we do not remove `&`.
    const envs = ["cases", "split"];

    let processedEquationString = "";
    // equationString is a line of LaTeX equation
    processedEquationString += "$$\n";
    processedEquationString += equationString
        .replace(/\\label\{[^}]*\}/g, "") // Remove \label{...}
        .split(
            new RegExp(
                `(\\\\begin\\{(?:[^}]*matrix[^}]*|${envs.join("|")})\\}[\\s\\S]*?\\\\end\\{(?:[^}]*matrix[^}]*|${envs.join("|")})\\})`,
            ),
        )
        .map((segment) => {
            // If the segment is inside one of the environments, do not remove the & symbols
            if (
                new RegExp(
                    `\\\\begin\\{(?:[^}]*matrix[^}]*|${envs.join("|")})\\}`,
                ).test(segment)
            ) {
                return segment;
            }
            // If the segment is outside, remove the & symbols
            return segment.replace(/&/g, "");
        })
        .join("");
    processedEquationString += "\n$$\n";
    return processedEquationString;
}
