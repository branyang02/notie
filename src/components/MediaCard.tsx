import {
  SiDjango,
  SiExpress,
  SiFastapi,
  SiFlask,
  SiGooglecloud,
  SiHeroku,
  SiJavascript,
  SiMeta,
  SiOpenai,
  SiPython,
  SiPytorch,
  SiReact,
  SiTypescript,
} from '@icons-pack/react-simple-icons';
import { Paper } from '@mui/material';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import * as React from 'react';
import { Link } from 'react-router-dom';

type MediaCardProps = {
  image?: string;
  gif?: string;
  title: string;
  description: string;
  button: string;
  link: string;
  techStack: string[];
};

function getLogo(tech: string) {
  switch (tech) {
    case 'React':
      return <SiReact color="#61DAFB" size={24} />;
    case 'TypeScript':
      return <SiTypescript color="#007ACC" size={24} />;
    case 'JavaScript':
      return <SiJavascript color="#F7DF1E" size={24} />;
    case 'Python':
      return <SiPython color="#3776AB" size={24} />;
    case 'Django':
      return <SiDjango color="#092E20" size={24} />;
    case 'OpenAI':
      return <SiOpenai color="#FF0080" size={24} />;
    case 'Google Cloud':
      return <SiGooglecloud color="#4285F4" size={24} />;
    case 'Flask':
      return <SiFlask color="#000000" size={24} />;
    case 'PyTorch':
      return <SiPytorch color="#EE4C2C" size={24} />;
    case 'Heroku':
      return <SiHeroku color="#430098" size={24} />;
    case 'FastAPI':
      return <SiFastapi color="#009688" size={24} />;
    case 'Express':
      return <SiExpress color="#000000" size={24} />;
    case 'Meta':
      return <SiMeta color="#000000" size={24} />;
    case 'LlaMA2':
      return <SiMeta color="#000000" size={24} />;
    default:
      return null;
  }
}

const MediaCard: React.FC<MediaCardProps> = ({
  image,
  gif,
  title,
  description,
  button,
  link,
  techStack,
}) => {
  return (
    <Card sx={{ maxWidth: 345 }}>
      <CardMedia sx={{ height: 140 }} image={image ? image : gif} title={title} />{' '}
      <CardContent>
        <Typography gutterBottom variant="h5" component="div">
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
        <div style={{ display: 'flex', flexWrap: 'wrap' }}>
          {techStack.map((tech, index) => (
            <Paper
              key={index}
              elevation={5}
              style={{
                margin: '5px',
                padding: '5px',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <Typography
                variant="body2"
                color="text.secondary"
                style={{ marginRight: '10px' }}
              >
                {tech}
              </Typography>
              {getLogo(tech)}
            </Paper>
          ))}
        </div>
      </CardContent>
      <CardActions>
        <Button size="small" component={Link} to={link}>
          {button}
        </Button>
      </CardActions>
    </Card>
  );
};

export default MediaCard;
