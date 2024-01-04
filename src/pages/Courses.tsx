// src/pages/About.js
import '../styles/Courses.css';

import React from 'react';

import OrgChartTree from '../components/CourseTree';

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
