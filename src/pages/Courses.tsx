// src/pages/About.js
import React from 'react';
import OrgChartTree from '../components/CourseTree';
import '../styles/Courses.css';

const Courses = () => {
  return (
    <div>
      {/* <h1>About Page</h1>
      <p>This is the about page of our application.</p> */}
      <OrgChartTree />
    </div>
  );
};

export default Courses;
