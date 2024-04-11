import './styles/App.css';

import { IconButton, LightbulbIcon, MoonIcon } from 'evergreen-ui';
import { useEffect, useState } from 'react';
import Container from 'react-bootstrap/Container';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

import Biography from './components/Biography';
import ClustrMapsWidget from './components/ClusterMapsWidget';
import Contact from './components/Contact';
import NavBar from './components/NavBar';
import News from './components/News';
import WorkHistory from './components/WorkHistory';
import About from './pages/About';
import Blog from './pages/blog/Blog';
import Sora from './pages/blog/blogs/Sora/Sora';
import Transformers from './pages/blog/blogs/Transformers/Transformers';
import Courses from './pages/Courses';
import CSO2 from './pages/notes/CSO2/cso2';
import Notes from './pages/notes/Notes';
import Projects from './pages/Projects';
import SpellingBee from './pages/spelling_bee/SpellingBee';

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const savedMode = localStorage.getItem('darkMode');
    return savedMode === 'true'
      ? true
      : new Date().getHours() >= 18 || new Date().getHours() < 6;
  });

  useEffect(() => {
    const footnotesTitle = document.querySelector('.footnotes h2');
    if (footnotesTitle) {
      footnotesTitle.innerHTML = '<strong>References</strong>';
    }

    document.body.classList.toggle('dark-mode', darkMode);
    localStorage.setItem('darkMode', darkMode.toString());
  }, [darkMode]);

  return (
    <Router>
      <div style={{ position: 'relative' }}>
        <IconButton
          height={56}
          icon={darkMode ? LightbulbIcon : MoonIcon}
          onClick={() => setDarkMode(!darkMode)}
          style={{
            position: 'fixed',
            bottom: '20px',
            left: '20px',
            zIndex: 1000,
          }}
        />
        <div className="nav-bar">
          <NavBar />
        </div>
      </div>
      <Container as="main" className="py-4 px-3 mx-auto custom-padding">
        <Routes>
          <Route
            path="/"
            element={
              <>
                <Biography />
                <WorkHistory />
                <News />
              </>
            }
          />
          <Route path="/courses" element={<Courses />} />
          <Route path="/about" element={<About />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="projects/spelling-bee" element={<SpellingBee />} />
          <Route path="/blog" element={<Blog />} />
          <Route path="/blog/sora" element={<Sora />} />
          <Route path="/blog/transformers" element={<Transformers />} />
          <Route path="/notes" element={<Notes />} />
          <Route path="/notes/cso2" element={<CSO2 />} />
        </Routes>
        <ClustrMapsWidget />
      </Container>
    </Router>
  );
}

export default App;
