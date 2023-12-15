import '../styles/NavBar.css';

import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import { Link } from 'react-router-dom';

import resumePDF from '../assets/resume.pdf';

function NavBar() {
  return (
    <>
      <Navbar style={{ paddingTop: '20px' }}>
        <Container>
          <Navbar.Brand as={Link} to="/" className="navbar-brand-custom">
            Brandon (Yifan) Yang
          </Navbar.Brand>
          <Nav className="justify-content-end">
            <Nav.Link as={Link} to="/">
              Home
            </Nav.Link>
            <Nav.Link as={Link} to="/courses" className="navbar-brand-custom">
              Relevant Coursework
            </Nav.Link>
            {/* <Nav.Link as={Link} to="/cv">
              Curriculum Vitae (CV)
            </Nav.Link> */}
            <Nav.Link as={Link} to="/projects" className="navbar-brand-custom">
              Projects
            </Nav.Link>
            <Nav.Link href={resumePDF} target="_blank" className="navbar-brand-custom">
              Resume
            </Nav.Link>
            {/* <Nav.Link as={Link} to="/about">
              About
            </Nav.Link> */}
            {/* <Nav.Link as={Link} to="/contact" className="navbar-brand-custom">
              Contact
            </Nav.Link> */}
          </Nav>
        </Container>
      </Navbar>
      <hr className="navbar-separator" />
    </>
  );
}

export default NavBar;
