import { Alert, Box, Button, CircularProgress } from '@mui/material';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import React, { useEffect, useState } from 'react';

type WordType = {
  word: string;
  origin: string;
  definition: string;
  example: string;
};

type Dictionary = {
  word: string;
};

type WordProps = {
  wordDictionary: Dictionary[];
};

const Word: React.FC<WordProps> = ({ wordDictionary }) => {
  const [dictionary, setDictionary] = useState<Dictionary[]>(wordDictionary);
  const [currentWord, setCurrentWord] = useState<WordType | null>(null);
  const [isBlurred, setIsBlurred] = useState(true);
  const [isEnd, setIsEnd] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [audioSrc, setAudioSrc] = useState(''); // State to hold audio source URL

  useEffect(() => {
    selectWord();
    console.log(currentWord);
  }, []);

  const selectWord = async () => {
    setLoading(true);
    if (dictionary.length > 0) {
      const randomIndex = Math.floor(Math.random() * dictionary.length);
      const selectedWord = dictionary[randomIndex];
      console.log('Choosing word:', selectedWord.word);
      try {
        const response = await fetch(
          `https://yang-website-backend-c3338735a47f.herokuapp.com/api/word-details/${selectedWord.word}`,
        );
        if (response.ok) {
          setError(null);
          const wordDetails = await response.json();
          setCurrentWord({
            word: selectedWord.word,
            origin: wordDetails.language_of_origin,
            definition: wordDetails.definition,
            example: wordDetails.example_usage,
          });
          setIsBlurred(true);
          // Removing the selected word from the dictionary array
          setDictionary((dictionary) =>
            dictionary.filter((_, index) => index !== randomIndex),
          );

          try {
            const ttsResponse = await fetch(
              // `https://yang-website-backend-c3338735a47f.herokuapp.com/api/text-to-speech`,
              `http://127.0.0.1:5000/api/text-to-speech`,
              {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  text: `Your word is: ${selectedWord.word}. Origin: ${wordDetails.language_of_origin}. Definition: ${wordDetails.definition}. Example: ${wordDetails.example_usage}`,
                }),
              },
            );

            if (ttsResponse.ok) {
              const blob = await ttsResponse.blob();
              const audioUrl = URL.createObjectURL(blob);
              setAudioSrc(audioUrl);
            } else {
              console.error('Failed to fetch speech');
              setError('Error fetching speech.');
            }
          } catch (error) {
            console.error('Error fetching speech:', error);
          }
        } else {
          // remove the word from the dictionary array
          setDictionary((dictionary) =>
            dictionary.filter((_, index) => index !== randomIndex),
          );
          console.error('Failed to fetch word details');
          setError('Error fetching word details.');
        }
      } catch (error) {
        console.error('Error fetching word details:', error);
      }
    } else {
      setIsEnd(true);
    }

    setLoading(false);
  };

  const handleBlurToggle = () => {
    setIsBlurred(!isBlurred);
  };
  return (
    <div>
      {loading ? (
        <Box sx={{ display: 'flex', marginBottom: 2 }}>
          <CircularProgress />
        </Box>
      ) : (
        <div style={{ filter: isBlurred ? 'blur(19px)' : 'none' }}>
          <h1>{currentWord?.word}</h1>
          <h6>Origin: {currentWord?.origin}</h6>
          <h6>Definition: {currentWord?.definition}</h6>
          <h6>Example: {currentWord?.example}</h6>
        </div>
      )}
      {audioSrc && (
        <audio controls src={audioSrc}>
          Your browser does not support the audio element.
        </audio>
      )}
      <Button variant="contained" onClick={selectWord} sx={{ mr: 3 }}>
        Next Word
      </Button>
      <Button variant="contained" onClick={handleBlurToggle}>
        {isBlurred ? 'Unblur Word' : 'Blur Word'}
      </Button>
      {error && <Alert severity="error">{error}</Alert>}
      <Dialog open={isEnd}>
        <DialogTitle>
          You reached the end of your dictionary. Congratulations!
        </DialogTitle>
      </Dialog>
    </div>
  );
};

export default Word;
