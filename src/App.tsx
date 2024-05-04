import './styles/App.css';

import { IconButton, LightbulbIcon, MoonIcon } from 'evergreen-ui';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

import Header from './components/Header';
import { useDarkMode } from './context/DarkModeContext';
import NoteCards from './pages/NoteCards';
import Notes from './pages/Notes';

const App = () => {
  const { darkMode, toggleDarkMode } = useDarkMode();

  return (
    <Router>
      <div style={{ position: 'relative' }}>
        <IconButton
          height={56}
          icon={darkMode ? LightbulbIcon : MoonIcon}
          onClick={() => toggleDarkMode()}
          style={{
            position: 'fixed',
            bottom: '20px',
            left: '20px',
            zIndex: 1000,
          }}
        />
        <Header />
      </div>
      <div className="main-content">
        <Routes>
          <Route path="/" element={<NoteCards />} />
          <Route path="/notes/:noteId" element={<Notes />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
