import { useEffect, useState } from "react";
import Notie from "../components/Notie";
import { Button, Pane, majorScale } from "evergreen-ui";

const modules = import.meta.glob("./markdown-files/tutorial.md", {
    query: "?raw",
    import: "default",
});

interface ToggleThemeButtonsProps {
    setTheme: (theme: string) => void;
}

const ToggleThemeButtons = ({ setTheme }: ToggleThemeButtonsProps) => {
    const themes = [
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

    return (
        <Pane display="flex" justifyContent="space-around">
            {themeButtons}
        </Pane>
    );
};

const App = () => {
    const [markdownContent, setMarkdownContent] = useState<string>("");
    const [theme, setTheme] = useState<string>("default");

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
