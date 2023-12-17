import '../styles/WorkHistory.css';

import React from 'react';

const WorkHistory = () => {
  return (
    <div className="container mt-5 mb-5">
      <div className="row">
        <div className="col-md-8">
          <h4>Experience</h4>
          <ul className="timeline">{renderListItems()}</ul>
        </div>
      </div>
    </div>
  );
};

const renderListItems = () => {
  const listItems = [
    {
      date: 'May 2022 - Current',
      text: (
        <span>
          Research Assistant at{' '}
          <a href="https://www.collabrobotics.com/">Collaborative Robotics Lab</a>
        </span>
      ),
    },
    {
      date: 'Oct 2023 - Current',
      text: (
        <span>
          Co-Founder & Software Developer at <a>Voy</a>
        </span>
      ),
    },
    {
      date: 'May 2023 - Aug 2023',
      text: (
        <span>
          AI/ML Research Intern at{' '}
          <a href="https://engineering.virginia.edu/link-lab">UVA Link Lab</a>
        </span>
      ),
    },
    {
      date: 'Jan 2023 - May 2023',
      text: (
        <span>
          Lab Lead TA for{' '}
          <a href="https://www.cs.virginia.edu/~jh2jf/courses/cs2130/spring2023/">
            Computer Systems Organization (UVA CS 2130)
          </a>
        </span>
      ),
    },
  ];

  return listItems.map((item, index) => (
    <li key={index}>
      <span className="timeline-date">{item.date}</span>
      <p>{item.text}</p>
    </li>
  ));
};

export default WorkHistory;
