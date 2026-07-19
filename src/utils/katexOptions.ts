/**
 * Protocols allowed for `\href` commands. KaTeX resolves the protocol via
 * `utils.protocolFromUrl` before invoking `trust`: it strips leading
 * whitespace/control characters, lowercases the scheme, uses `"_relative"`
 * for relative and fragment-only URLs (e.g. Notie's own `#pre-eqn-...`
 * references), and rejects malformed schemes (returning `false` before
 * `trust` is ever called).
 */
const ALLOWED_HREF_PROTOCOLS = ["http", "https", "mailto", "_relative"];

/**
 * Color used by KaTeX to render invalid LaTeX source when `throwOnError` is
 * disabled. A distinct red keeps broken math visible during authoring
 * without blowing up the whole page render.
 */
const KATEX_ERROR_COLOR = "#cc0000";

export const katexOptions = {
    // Never throw on malformed LaTeX: a single bad equation should render
    // as visibly-flagged red source text instead of crashing the entire
    // markdown render tree.
    throwOnError: false,
    errorColor: KATEX_ERROR_COLOR,
    macros: {
        "\\eqref": "\\href{\\#pre-eqn-eqref:#1}{}",
        "\\ref": "\\href{\\#pre-eqn-ref:#1}{}",
        "\\label": "\\htmlId{#1}{}",
    },
    strict: false,
    trust: (context: { command: string; protocol?: string }) => {
        if (context.command === "\\htmlId") {
            return true;
        }
        if (context.command === "\\href") {
            return (
                typeof context.protocol === "string" &&
                ALLOWED_HREF_PROTOCOLS.includes(context.protocol)
            );
        }
        return false;
    },
};
