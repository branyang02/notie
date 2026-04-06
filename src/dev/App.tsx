import { useEffect, useState } from "react";
import Notie from "../components/Notie";
import { Pane, majorScale, Select } from "evergreen-ui";
import { NotieThemes } from "../config/NotieConfig";

const modules = import.meta.glob("./test.md", {
    query: "?raw",
    import: "default",
});

const THEMES: NotieThemes[] = [
    "default",
    "default dark",
    "Starlit Eclipse",
    "Starlit Eclipse Light",
];

const BACKGROUND: Record<NotieThemes, string> = {
    default: "#fff",
    "default dark": "#333",
    "Starlit Eclipse": "rgb(3 7 18)",
    "Starlit Eclipse Light": "#fff",
};

const App = () => {
    const [markdownContent, setMarkdownContent] = useState<string>("");
    const [theme, setTheme] = useState<NotieThemes>("default");

    useEffect(() => {
        const fetchNote = async () => {
            for (const path in modules) {
                const markdown = await modules[path]();
                const rawMDString = markdown as string;
                setMarkdownContent(rawMDString);
                break;
            }
        };

        fetchNote();
    }, []);

    return (
        <Pane background={BACKGROUND[theme]}>
            <Pane position="fixed" top={12} right={16} zIndex={1000}>
                <Select
                    value={theme}
                    onChange={(e) => setTheme(e.target.value as NotieThemes)}
                >
                    {THEMES.map((t) => (
                        <option key={t} value={t}>
                            {t}
                        </option>
                    ))}
                </Select>
            </Pane>
            <Pane
                maxWidth={majorScale(180)}
                padding={20}
                style={{
                    margin: "0 auto",
                }}
            >
                <Notie
                    markdown={markdownContent}
                    theme={theme}
                    config={{
                        showTableOfContents: true,
                        fontSize: "1.1em",
                        theme: {
                            fontFamily: '"Computer Modern Serif", serif',
                            customFontUrl:
                                "https://cdn.jsdelivr.net/gh/bitmaks/cm-web-fonts@latest/fonts.css",
                            blockquoteStyle: "latex",
                            numberedHeading: true,
                            tocMarker: false,
                        },
                    }}
                />
            </Pane>
        </Pane>
    );
};

export default App;
