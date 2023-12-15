import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';
import { Box, IconButton, LinearProgress } from '@mui/material';
import React from 'react';

type ScoreCounterProps = {
  participantName: string;
  score: number;
  updateScore: (name: string, score: number) => void;
};

const ScoreCounter: React.FC<ScoreCounterProps> = ({
  participantName,
  score,
  updateScore,
}) => {
  const incrementScore = () => {
    updateScore(participantName, Math.min(score + 10, 100));
  };

  const decrementScore = () => {
    updateScore(participantName, Math.max(score - 5, -100));
  };

  const progress = (score + 100) / 2;

  return (
    <Box sx={{ mb: 4 }}>
      <Box sx={{ mb: 2 }}>
        {participantName}: {score}
      </Box>
      <LinearProgress variant="determinate" value={progress} />
      <Box sx={{ mt: 2 }}>
        <IconButton onClick={decrementScore} color="error" sx={{ mr: 1 }}>
          <RemoveIcon />
        </IconButton>
        <IconButton onClick={incrementScore} color="success">
          <AddIcon />
        </IconButton>
      </Box>
    </Box>
  );
};

export default ScoreCounter;
