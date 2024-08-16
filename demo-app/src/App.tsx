import { Pane, majorScale } from "evergreen-ui";
import { Notie } from "notie-markdown";
import NavBar from "./components/NavBar";
import { Route, Routes } from "react-router-dom";
import Examples from "./pages/Examples";
import ExamplePage from "./pages/ExamplePage";
import { useEffect, useState } from "react";
import { useTheme } from "./context/useTheme";
import MyChart from "./components/MyChart";

const homeModule = import.meta.glob("../../README.md", {
    query: "?raw",
    import: "default",
});

const contributionModule = import.meta.glob("../../CONTRIBUTING.md", {
    query: "?raw",
    import: "default",
});

const tutorialModule = import.meta.glob("./pages/tutorial.md", {
    query: "?raw",
    import: "default",
});

const App = () => {
    const [homeContent, setHomeContent] = useState<string>("");
    const [tutorialContent, setTutorialContent] = useState<string>("");
    const [contributionContent, setContributionContent] = useState<string>("");
    const { theme } = useTheme();
    const darkMode = theme === "default dark" || theme === "Starlit Eclipse";

    useEffect(() => {
        async function fetchContent() {
            const newhomeContent = (await homeModule[
                Object.keys(homeModule)[0]
            ]()) as string;
            const newtutorialContent = (await tutorialModule[
                Object.keys(tutorialModule)[0]
            ]()) as string;
            const newcontributionContent = (await contributionModule[
                Object.keys(contributionModule)[0]
            ]()) as string;

            setHomeContent(newhomeContent);
            setTutorialContent(newtutorialContent);
            setContributionContent(newcontributionContent);
        }

        fetchContent();
    }, []);

    const customComponents = {
        myChart: () => <MyChart darkMode={darkMode} />,
    };

    function getBackgroundColor() {
        switch (theme) {
            case "default dark":
                return "#333";
            case "Starlit Eclipse":
                return "rgb(3 7 18)";
            case "Starlit Eclipse Light":
                return "rgb(255 255 255)";
            default:
                return "#fff";
        }
    }

    return (
        <Pane
            backgroundColor={getBackgroundColor()}
            style={{
                display: "flex",
                flexDirection: "column",
                minHeight: "100vh",
            }}
        >
            <NavBar />

            <Pane display="flex" alignItems="center" justifyContent="center">
                <Pane
                    display="flex"
                    alignItems="center"
                    justifyContent="space-between"
                    width="100%"
                    maxWidth={majorScale(180)}
                    padding={majorScale(3)}
                >
                    <Routes>
                        <Route
                            path="/"
                            element={
                                <Notie markdown={homeContent} theme={theme} />
                            }
                        />
                        <Route path="/examples" element={<Examples />} />
                        <Route
                            path="/tutorial"
                            element={
                                <Notie
                                    markdown={tutorialContent}
                                    theme={theme}
                                    customComponents={customComponents}
                                />
                            }
                        />
                        <Route
                            path="/contribute"
                            element={
                                <Notie
                                    markdown={contributionContent}
                                    theme={theme}
                                />
                            }
                        />
                        <Route
                            path="/examples/:noteId"
                            element={<ExamplePage />}
                        />
                    </Routes>
                </Pane>
            </Pane>
        </Pane>
    );
};

export default App;
