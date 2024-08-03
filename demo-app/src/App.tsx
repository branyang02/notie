import { Pane, majorScale } from "evergreen-ui";
import { Notie } from "notie-markdown";
import NavBar from "./components/NavBar";
import { useDarkMode } from "./context/DarkModeContext";
import { Route, Routes } from "react-router-dom";
import Examples from "./pages/Examples";
import ExamplePage from "./pages/ExamplePage";

const homeModule = import.meta.glob("./pages/home.md", {
  query: "?raw",
  import: "default",
});

const HOMECONTENT = (await homeModule[Object.keys(homeModule)[0]]()) as string;

const tutorialModule = import.meta.glob("./pages/tutorial.md", {
  query: "?raw",
  import: "default",
});

const TUTORIALCONTENT = (await tutorialModule[
  Object.keys(tutorialModule)[0]
]()) as string;

const App = () => {
  const { darkMode } = useDarkMode();

  return (
    <Pane
      background={darkMode ? "#333" : "white"}
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
              element={<Notie markdown={HOMECONTENT} darkMode={darkMode} />}
            />
            <Route path="/examples" element={<Examples />} />
            <Route
              path="/tutorial"
              element={<Notie markdown={TUTORIALCONTENT} darkMode={darkMode} />}
            />
            <Route path="/examples/:noteId" element={<ExamplePage />} />
          </Routes>
        </Pane>
      </Pane>
    </Pane>
  );
};

export default App;
