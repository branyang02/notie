import { Button } from '@mui/material';
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

const rnd_2 = [
  {
    word: 'Connoisseur',
    origin: 'French',
    definition:
      'A person who is an expert in a particular area, especially when it comes to appreciating or identifying fine arts, food, or wines.',
    example:
      "When it came to instant noodles, he proudly declared himself a connoisseur, saying, 'I can tell a good noodle from a mediocre one just by the sound of the microwave!'",
  },
];

const Word: React.FC<WordProps> = ({ wordDictionary }) => {
  console.log(wordDictionary);
  const [dictionary, setDictionary] = useState<WordType[]>(rnd_2);
  const [currentWord, setCurrentWord] = useState<WordType | null>(null);
  const [isBlurred, setIsBlurred] = useState(true);
  const [isEnd, setIsEnd] = useState(false);

  useEffect(() => {
    selectWord();
  }, []);

  const selectWord = () => {
    if (dictionary.length > 0) {
      const randomIndex = Math.floor(Math.random() * dictionary.length);
      const selectedWord = dictionary[randomIndex];
      setCurrentWord(selectedWord);
      console.log(
        `Word: ${selectedWord.word}, \nOrigin: ${selectedWord.origin}, \nDefinition: ${selectedWord.definition}, \nExample: ${selectedWord.example}`,
      );
      setDictionary(dictionary.filter((_, index) => index !== randomIndex));
      setIsBlurred(true);
    } else {
      setIsEnd(true);
    }
  };

  const handleBlurToggle = () => {
    setIsBlurred(!isBlurred);
  };
  return (
    <div>
      <div style={{ filter: isBlurred ? 'blur(19px)' : 'none' }}>
        <h1>{currentWord?.word}</h1>
      </div>
      <Button variant="contained" onClick={selectWord} sx={{ mr: 3 }}>
        Next Word
      </Button>
      <Button variant="contained" onClick={handleBlurToggle}>
        {isBlurred ? 'Unblur Word' : 'Blur Word'}
      </Button>
      <Dialog open={isEnd}>
        <DialogTitle>
          You reached the end of your dictionary. Congratulations!
        </DialogTitle>
      </Dialog>
    </div>
  );
};

export default Word;
