export const katexOptions = {
    macros: {
        "\\eqref": "\\href{\\#pre-eqn-eqref:#1}{}",
        "\\ref": "\\href{\\#pre-eqn-ref:#1}{}",
        "\\label": "\\htmlId{#1}{}",
    },
    strict: false,
    trust: (context: { command: string }) =>
        ["\\htmlId", "\\href"].includes(context.command),
};
