import '../styles/Biography.css';

import React from 'react';

import profileImage from '../assets/Brandon_Yang.jpg';
import Contact from './Contact';

const Biography: React.FC = () => {
  return (
    <div className="biography-container">
      <div className="biography-text">
        <h3>Biography</h3>
        <p>
          I am a third-year B.S. Computer Science student at the{' '}
          <a href="https://engineering.virginia.edu/"> University of Virginia</a>. I am
          interested in Machine Learning (ML), Reinforcement Learning (RL), Computer
          Vision, Robotics, and Software Development . I am currently working as a
          research assistant with{' '}
          <a href="https://engineering.virginia.edu/faculty/tariq-iqbal">
            Professor Iqbal
          </a>{' '}
          at the <a href="https://www.collabrobotics.com/">Collaborative Robotics Lab</a>{' '}
          at UVA. I am particularly interested in applying ML and RL to applications
          robotics, as well as building software applications using AI components such as
          LLM. I am also interested in the intersection between ML and Computer Vision,
          and I am currently working on a project that uses LLM and Diffusion Models to
          generate goal images for Imitation and Reinforcement Learning.
        </p>
        <Contact />
      </div>
      <div className="biography-image">
        <img src={profileImage} alt="Brandon Yang" />
      </div>
    </div>
  );
};

export default Biography;
