/**
 * Protocols allowed for `\href` commands. KaTeX resolves the protocol via
 * `utils.protocolFromUrl` before invoking `trust`: it strips leading
 * whitespace/control characters, lowercases the scheme, uses `"_relative"`
 * for relative and fragment-only URLs (e.g. Notie's own `#pre-eqn-...`
 * references), and rejects malformed schemes (returning `false` before
 * `trust` is ever called).
 */
const ALLOWED_HREF_PROTOCOLS = ["http", "https", "mailto", "_relative"];

export const katexOptions = {
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
