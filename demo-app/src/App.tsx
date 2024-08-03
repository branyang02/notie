import { Pane, majorScale } from 'evergreen-ui';
import { Notie } from 'notie-markdown';
import NavBar from './components/NavBar';
import { useDarkMode } from './context/DarkModeContext';
import { Route, Routes } from 'react-router-dom';
import Examples from './pages/Examples';
import ExamplePage from './pages/ExamplePage';
import { useEffect, useState } from 'react';

const homeModule = import.meta.glob('./pages/home.md', {
  query: '?raw',
  import: 'default',
});

const tutorialModule = import.meta.glob('./pages/tutorial.md', {
  query: '?raw',
  import: 'default',
});

const App = () => {
  const { darkMode } = useDarkMode();
  const [homeContent, setHomeContent] = useState<string>('');
  const [tutorialContent, setTutorialContent] = useState<string>('');

  useEffect(() => {
    async function fetchContent() {
      const newhomeContent = (await homeModule[Object.keys(homeModule)[0]]()) as string;
      const newtutorialContent = (await tutorialModule[
        Object.keys(tutorialModule)[0]
      ]()) as string;

      setHomeContent(newhomeContent);
      setTutorialContent(newtutorialContent);
    }

    fetchContent();
  }, []);

  return (
    <Pane
      background={darkMode ? '#333' : 'white'}
      style={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
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
              element={<Notie markdown={homeContent} darkMode={darkMode} />}
            />
            <Route path="/examples" element={<Examples />} />
            <Route
              path="/tutorial"
              element={<Notie markdown={tutorialContent} darkMode={darkMode} />}
            />
            <Route path="/examples/:noteId" element={<ExamplePage />} />
          </Routes>
        </Pane>
      </Pane>
    </Pane>
  );
};

export default App;
