import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './styles/App.css';
import Container from 'react-bootstrap/Container';
import Biography from './components/Biography';
import NavBar from './components/NavBar';
import News from './components/News';
import WorkHistory from './components/WorkHistory';
import Courses from './pages/Courses';
import About from './pages/About';
import Contact from './components/Contact';

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
          {/* <Route path='/cv' element={<CV />} />
          <Route path='/resume' element={<Resume />} /> */}
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          {/* Add other routes here */}
        </Routes>
      </Container>
    </Router>
  );
}

export default App;
