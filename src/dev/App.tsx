import { useEffect, useState } from "react";
import Notie from "../components/Notie";
import { Button, Pane, majorScale } from "evergreen-ui";
import { NotieThemes } from "../utils/useNotieConfig";

const modules = import.meta.glob("./markdown-files/tutorial.md", {
    query: "?raw",
    import: "default",
});

const ToggleThemeButtons = ({
    setTheme,
}: {
    setTheme: (theme: NotieThemes) => void;
}) => {
    const themes: NotieThemes[] = [
        "default",
        "default dark",
        "Starlit Eclipse",
        "Starlit Eclipse Light",
    ];

    const themeButtons = themes.map((theme) => (
        <Button key={theme} onClick={() => setTheme(theme)}>
            {theme}
        </Button>
    ));

    return <Pane display="flex">{themeButtons}</Pane>;
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

    const customComponents = {
        ToggleThemeButtons: () => <ToggleThemeButtons setTheme={setTheme} />,
    };

    return (
        <Pane background="#fff">
            <Pane
                maxWidth={majorScale(180)}
                padding={20}
                style={{
                    margin: "0 auto",
                }}
            >
                <Notie
                    markdown={markdownContent}
                    config={{
                        showTableOfContents: true,
                    }}
                    theme={theme}
                    customComponents={customComponents}
                />
            </Pane>
        </Pane>
    );
};

export default App;
