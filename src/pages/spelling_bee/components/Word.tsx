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

const rnd_2 = [
  {
    word: 'Connoisseur',
    origin: 'French',
    definition:
      'A person who is an expert in a particular area, especially when it comes to appreciating or identifying fine arts, food, or wines.',
    example:
      "When it came to instant noodles, he proudly declared himself a connoisseur, saying, 'I can tell a good noodle from a mediocre one just by the sound of the microwave!'",
  },
  {
    word: 'Krankenwagen',
    origin: 'German',
    definition:
      "The German word for 'ambulance,' used for transporting sick or injured people to the hospital.",
    example:
      "After a minor mishap, he jokingly said, 'I had to call the krankenwagen. Turns out, even my bruises have a strong German accent!'",
  },
  {
    word: 'Reykjavik',
    origin: 'Icelandic',
    definition:
      'The capital and largest city of Iceland, known for its stunning landscapes, geothermal springs, and vibrant culture.',
    example:
      "He chuckled and said, 'Iceland has Reykjavik, a city that's so cool, it's like the Arctic's answer to Las Vegas, but with fewer slot machines and more icebergs!'",
  },
  {
    word: 'Rhinorrhagia',
    origin: 'Greek',
    definition:
      'A medical term for a severe and prolonged nosebleed or nasal hemorrhage.',
    example:
      "He chuckled and said, 'Rhinorrhagia is like a nosebleed's way of making an entrance. It's the grand finale of sneezing!'",
  },
  {
    word: 'Beethovenian',
    origin: 'German (inspired by Ludwig van Beethoven)',
    definition:
      'Relating to or characteristic of the renowned composer Ludwig van Beethoven or his music.',
    example:
      "She chuckled and said, 'When it comes to air guitar, he's truly Beethovenian. He turns every living room into a symphony hall!'",
  },
  {
    word: 'Kaffeeklatsch',
    origin: 'German',
    definition: 'A social gathering where people chat and gossip over coffee and cake.',
    example:
      "He chuckled and said, 'A Kaffeeklatsch is like a caffeine-fueled therapy session. We spill the beans while sipping the coffee!'",
  },
];

const Word: React.FC = () => {
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
