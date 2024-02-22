import './styles/App.css';

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
import Sora from './pages/blog/blogs/Sora';
import Transformers from './pages/blog/blogs/Transformers';
import Courses from './pages/Courses';
import Projects from './pages/Projects';
import SpellingBee from './pages/spelling_bee/SpellingBee';

function App() {
  return (
    <Router>
      <NavBar />
      <Container as="main" className="py-4 px-3 mx-auto">
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
        </Routes>
        <ClustrMapsWidget />
      </Container>
    </Router>
  );
}

export default App;
