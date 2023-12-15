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

type Dictionary = {
  word: string;
};

type MenuProps = {
  onStart: (players: Player[], dictionary: Dictionary[]) => void;
};

const Menu: React.FC<MenuProps> = ({ onStart }) => {
  const [name, setName] = useState<string>('');
  const [players, setPlayers] = useState<Player[]>([]);
  const [word, setWord] = useState<string>('');
  const [dictionary, setDictionary] = useState<Dictionary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [wordError, setWordError] = useState<string | null>(null);

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

  const addWord = (
    event: MouseEvent<HTMLButtonElement> | React.FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();
    if (word.trim() !== '') {
      if (dictionary.some((dictWord) => dictWord.word === word)) {
        setWordError('Word already exists');
      } else {
        setDictionary([...dictionary, { word }]);
        setWord('');
        setWordError(null);
      }
    } else {
      setWordError('Word cannot be blank');
    }
  };

  const deletePlayer = (index: number) => {
    setPlayers(players.filter((_, i) => i !== index));
  };

  const deleteWord = (index: number) => {
    setDictionary(dictionary.filter((_, i) => i !== index));
  };

  const handleNameChange = (event: ChangeEvent<HTMLInputElement>) => {
    setName(event.target.value);
  };

  const handleWordChange = (event: ChangeEvent<HTMLInputElement>) => {
    setWord(event.target.value);
  };

  const handleStart = (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    onStart(players, dictionary);
  };

  return (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      style={{ minHeight: '100vh' }}
    >
      <h1>😈 Welcome to Spelling Bee 😈</h1>
      <Grid container spacing={2}>
        <Grid item xs={6}>
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
        </Grid>
        <Grid item xs={6}>
          {wordError && <Alert severity="error">{wordError}</Alert>}
          <form onSubmit={addWord}>
            <Grid container spacing={1} alignItems="center" justifyContent="center">
              <Grid item xs={6}>
                <TextField
                  value={word}
                  onChange={handleWordChange}
                  label="Enter Word"
                  variant="outlined"
                  fullWidth
                />
              </Grid>
              <Grid item xs={6}>
                <Button variant="contained" type="submit" fullWidth>
                  Enter Word
                </Button>
              </Grid>
            </Grid>
          </form>

          <List>
            {dictionary.map((word, index) => (
              <ListItem key={index}>
                <ListItemText primary={word.word} />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={() => deleteWord(index)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </Grid>
      </Grid>
      <Button variant="contained" color="success" onClick={handleStart}>
        Start Game
      </Button>
    </Box>
  );
};

export default Menu;
