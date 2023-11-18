import React from 'react';

const newsData = [
  {
    date: '11/2023',
    news: "Voy received $1,000 in funding from winner UVA's Entrepreneurship Cup!",
  },
  {
    date: '11/2023',
    news: "Presented GLOMA at UVA's Fall Research Expo!",
  },
  {
    date: '09/2023',
    news: (
      <span>
        Invited to present GLOMA at UVA's{' '}
        <a href="https://engineering.virginia.edu/advancement/ways-give/thornton-society">
          Thornton Society
        </a>{' '}
        Dinner!
      </span>
    ),
  },
  {
    date: '10/2023',
    news: "Co-founded Voy, a startup that's building a platform for non-profit organizations to manage their volunteers.",
  },
  {
    date: '09/2023',
    news: (
      <span>
        Received 3rd place overall at{' '}
        <a href="https://vthacks-11.devpost.com/project-gallery">VTHacks 11</a> for
        building <a href="https://github.com/ewei2406/SmartOH">Smart OH</a>, a smart
        office hour queue system with AI and NLP integration!
      </span>
    ),
  },
  {
    date: '07/2023',
    news: "Presented poster GLOMA: Grounded Location for Object Manipulation at UVA's Summer Research Symposium!",
  },
  {
    date: '06/2023',
    news: 'Started working at UVA Link Lab with support from UVA Engineering Undergraduate Research Program!',
  },
  {
    date: '03/2023',
    news: "Accepted to UVA Dean's Engineering Undergraduate Research Program for Summer 2023!",
  },
  {
    date: '01/2023',
    news: 'Started working as a Lab Lead TA for CS 2130: Computer Systems & Organizations at UVA!',
  },
  {
    date: '11/2022',
    news: 'Demoed robot grasping system using Fetch Robot and AprilTag at UVA Engineering Open House.',
  },
  {
    date: '09/2022',
    news: (
      <span>
        Developed{' '}
        <a href="https://github.com/branyang02/Panda_Robot_AprilTag">
          object localization tool
        </a>{' '}
        for Panda Robot using AprilTag, ROS, and OpenCV.
      </span>
    ),
  },
  {
    date: '07/2022',
    news: (
      <span>
        Invited to work with high schooler as part of{' '}
        <a href="https://summer.virginia.edu/uva-advance">UVA Advance</a> program.
        Developed room mapping navigation system with Double Robot using Python and ROS.
      </span>
    ),
  },
  {
    date: '06/2022',
    news: "Started working as a Research Assistant at UVA's Collaborative Robotics Lab!",
  },
  {
    date: '08/2021',
    news: 'Started B.S. in Computer Science at the University of Virginia.',
  },
];

const News: React.FC = () => {
  return (
    <div>
      <h3>News</h3>
      {newsData.map((item, index) => (
        <div key={index}>
          <p>
            {item.date}: {item.news}
          </p>
        </div>
      ))}
    </div>
  );
};

export default News;
