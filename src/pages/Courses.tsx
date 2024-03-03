import '../styles/Courses.css';

import { useEffect } from 'react';

import OrgChartTree from '../components/CourseTree';

const Courses = () => {
  useEffect(() => {
    document.body.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = '';
    };
  }, []);
  return (
    <div className="full-padding">
      <OrgChartTree />
    </div>
  );
};

export default Courses;
