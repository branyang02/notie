import { useEffect, useState } from "react";
import Notie from "../components/Notie";
import { Pane, Switch, Heading } from "evergreen-ui";

const modules = import.meta.glob("./markdown-files/ml.md", {
    query: "?raw",
    import: "default",
});

const App = () => {
    const [markdownContent, setMarkdownContent] = useState<string>("");
    const [darkMode, setDarkMode] = useState<boolean>(false);

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

    const toggleDarkMode = () => {
        setDarkMode((prevMode) => !prevMode);
    };

    return (
        <Pane
            maxWidth={1500}
            padding={20}
            style={{
                margin: "0 auto",
            }}
        >
            <Pane
                display="flex"
                justifyContent="space-between"
                alignItems="center"
                marginBottom={20}
            >
                <Pane
                    display="flex"
                    alignItems="center"
                    justifyContent="space-between"
                >
                    <Heading size={700}>Dark Mode Switch</Heading>
                    <Switch
                        height={24}
                        checked={darkMode}
                        onChange={toggleDarkMode}
                    />
                </Pane>
            </Pane>
            <Notie markdown={markdownContent} darkMode={darkMode} />
        </Pane>
    );
};

export default App;
