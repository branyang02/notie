import { useEffect, useState } from "react";
import Notie from "../components/Notie";
import { Pane, majorScale } from "evergreen-ui";

const modules = import.meta.glob("./test.md", {
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
                    // theme={"Starlit Eclipse"}
                />
            </Pane>
        </Pane>
    );
};

export default App;
