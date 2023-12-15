import { Grid } from '@mui/material';
import { useState } from 'react';

import Leaderboard from './components/Leaderboard';
import Menu from './components/Menu';
import ScoreCounter from './components/ScoreCounter';
import Word from './components/Word';

type Player = {
  name: string;
  score: number;
};

function App() {
  const [gameStarted, setGameStarted] = useState(false);
  const [players, setPlayers] = useState<Player[]>([]);

  const startGame = (players: Player[]) => {
    setPlayers(players);
    setGameStarted(true);
  };

  if (!gameStarted) {
    return <Menu onStart={startGame} />;
  }

  const updateScore = (name: string, score: number) => {
    setPlayers((players) =>
      players.map((player) => (player.name === name ? { ...player, score } : player)),
    );
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={8}>
        <Grid container spacing={2}>
          {players.map((player) => (
            // eslint-disable-next-line react/jsx-key
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <ScoreCounter
                key={player.name}
                participantName={player.name}
                score={player.score}
                updateScore={updateScore}
              />
            </Grid>
          ))}
        </Grid>
      </Grid>
      <Grid item xs={12} md={4}>
        <Leaderboard players={players} />
        <Word />
      </Grid>
    </Grid>
  );
}

export default App;
