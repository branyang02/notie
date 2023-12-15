import React from 'react';

type Player = {
  name: string;
  score: number;
};

type LeaderboardProps = {
  players: Player[];
};

const Leaderboard: React.FC<LeaderboardProps> = ({ players }) => {
  const sortedPlayers = [...players].sort((a, b) => b.score - a.score);
  const topPlayers = sortedPlayers.slice(0, 5);
  const lastPlayers = sortedPlayers.slice(-5);

  return (
    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
      <div>
        <h2>Top 5 Players</h2>
        {topPlayers.map((player, index) => (
          <p key={index}>
            {index + 1}. {player.name}: {player.score}
          </p>
        ))}
      </div>
      <div>
        <h2>Last 5 Players</h2>
        {lastPlayers.map((player, index) => (
          <p key={index}>
            {index + 1}. {player.name}: {player.score}
          </p>
        ))}
      </div>
    </div>
  );
};

export default Leaderboard;
