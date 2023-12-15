import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import * as React from 'react';
import { Link } from 'react-router-dom';

type MediaCardProps = {
  image: string;
  title: string;
  description: string;
  button: string;
  link: string;
};

const MediaCard: React.FC<MediaCardProps> = ({
  image,
  title,
  description,
  button,
  link,
}) => {
  return (
    <Card sx={{ maxWidth: 345 }}>
      <CardMedia sx={{ height: 140 }} image={image} title={title} />
      <CardContent>
        <Typography gutterBottom variant="h5" component="div">
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
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
