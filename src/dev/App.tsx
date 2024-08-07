import { useEffect, useState } from "react";
import Notie from "../components/Notie";
import { Pane } from "evergreen-ui";

const modules = import.meta.glob("./markdown-files/cso2.md", {
    query: "?raw",
    import: "default",
});

const App = () => {
    const [markdownContent, setMarkdownContent] = useState<string>("");

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
        <Pane
            maxWidth={1500}
            padding={20}
            style={{
                margin: "0 auto",
            }}
        >
            <Notie markdown={markdownContent} darkMode={false} />
        </Pane>
    );
};

export default App;
