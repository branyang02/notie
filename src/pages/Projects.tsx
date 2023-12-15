import Grid from '@mui/material/Grid';
import React from 'react';

import spellingBeeImage from '../assets/spelling_bee.png';
import MediaCard from '../components/MediaCard';

const Projects: React.FC = () => {
  const projects = [
    {
      image: spellingBeeImage,
      title: 'Spelling Bee',
      description:
        'Fun Party Spelling Bee Game with score counter and word list. Made with React and TypeScript.',
      button: 'Try Live',
      link: '/projects/spelling-bee',
    },
    // TODO: more projects
  ];

  return (
    <div>
      <h1>Projects</h1>
      <Grid container spacing={2}>
        {projects.map((project, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <MediaCard
              image={project.image}
              title={project.title}
              description={project.description}
              button={project.button}
              link={project.link}
            />
          </Grid>
        ))}
      </Grid>
    </div>
  );
};

export default Projects;
