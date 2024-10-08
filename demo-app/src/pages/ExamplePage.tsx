import { Notie } from "notie-markdown";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Pane, Spinner } from "evergreen-ui";
import { useTheme } from "../context/useTheme";

const modules = import.meta.glob("../assets/**.md", {
    query: "?raw",
    import: "default",
});

const ExamplePage = () => {
    const [markdownContent, setMarkdownContent] = useState<string>("");
    const { noteId } = useParams();
    const [loading, setLoading] = useState(true);
    const { theme } = useTheme();

    useEffect(() => {
        const fetchNote = async () => {
            if (!noteId) return;

            for (const path in modules) {
                if (path.includes(noteId)) {
                    const markdown = await modules[path]();
                    const rawMDString = markdown as string;
                    setMarkdownContent(rawMDString);
                    break;
                }
            }
            setLoading(false);
        };

        fetchNote();
    }, [noteId]);

    if (loading) {
        return (
            <Pane
                display="flex"
                alignItems="center"
                justifyContent="center"
                style={{
                    margin: "0 auto",
                }}
            >
                <Spinner />
            </Pane>
        );
    }

    return <Notie markdown={markdownContent} theme={theme} />;
};

export default ExamplePage;
