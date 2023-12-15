import DeleteIcon from '@mui/icons-material/Delete';
import {
  Alert,
  Box,
  Button,
  Grid,
  IconButton,
  List,
  ListItem,
  ListItemSecondaryAction,
  ListItemText,
  TextField,
} from '@mui/material';
import React, { ChangeEvent, MouseEvent, useState } from 'react';

type Player = {
  name: string;
  score: number;
};

type MenuProps = {
  onStart: (players: Player[]) => void;
};

const Menu: React.FC<MenuProps> = ({ onStart }) => {
  const [name, setName] = useState<string>('');
  const [players, setPlayers] = useState<Player[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [newWord, setNewWord] = useState<string>('');

  const addPlayer = (
    event: MouseEvent<HTMLButtonElement> | React.FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();
    if (name.trim() !== '') {
      if (players.some((player) => player.name === name)) {
        setError('Name already exists');
      } else {
        setPlayers([...players, { name, score: 0 }]);
        setName('');
        setError(null);
      }
    } else {
      setError('Name cannot be blank');
    }
  };

  const handleWordChange = (event: ChangeEvent<HTMLInputElement>) => {
    setNewWord(event.target.value);
  };

  const removeWord = (wordToRemove: string) => {
    setDictionary(dictionary.filter((word) => word !== wordToRemove));
  };

  const deletePlayer = (index: number) => {
    setPlayers(players.filter((_, i) => i !== index));
  };

  const handleNameChange = (event: ChangeEvent<HTMLInputElement>) => {
    setName(event.target.value);
  };

  const handleStart = (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    onStart(players);
  };

  return (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      style={{ minHeight: '100vh' }}
    >
      <h1>😈 Welcome to Spelling Bee 😈</h1>
      {error && <Alert severity="error">{error}</Alert>}
      <form onSubmit={addPlayer}>
        <Grid container spacing={1} alignItems="center" justifyContent="center">
          <Grid item xs={6}>
            <TextField
              value={name}
              onChange={handleNameChange}
              label="Enter User"
              variant="outlined"
              fullWidth
            />
          </Grid>
          <Grid item xs={6}>
            <Button variant="contained" type="submit" fullWidth>
              Enter User
            </Button>
          </Grid>
        </Grid>
      </form>

      <List>
        {players.map((player, index) => (
          <ListItem key={index}>
            <ListItemText primary={player.name} />
            <ListItemSecondaryAction>
              <IconButton
                edge="end"
                aria-label="delete"
                onClick={() => deletePlayer(index)}
              >
                <DeleteIcon />
              </IconButton>
            </ListItemSecondaryAction>
          </ListItem>
        ))}
      </List>
      <Button variant="contained" color="success" onClick={handleStart}>
        Start Game
      </Button>
    </Box>
  );
};

export default Menu;
