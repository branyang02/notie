import '../../styles/blogPost.css';

import {
  Card,
  Heading,
  IconButton,
  LightbulbIcon,
  Link,
  majorScale,
  MoonIcon,
  Pane,
  Text,
} from 'evergreen-ui';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Blog() {
  const navigate = useNavigate();
  const [darkMode, setDarkMode] = useState(
    new Date().getHours() >= 18 || new Date().getHours() < 6,
  );

  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

  const blogPosts = [
    {
      title: `A Deep Dive into OpenAI's Sora`,
      date: 'February 20, 2024',
      link: `/blog/sora`,
    },
    {
      title: 'Transformers',
      date: 'February 10, 2024',
      link: `/blog/transformers`,
    },
  ];

  const handleCardClick = (path: string) => {
    navigate(path);
  };

  return (
    <div style={{ position: 'relative' }}>
      <IconButton
        height={56}
        icon={darkMode ? LightbulbIcon : MoonIcon}
        onClick={() => setDarkMode(!darkMode)}
        style={{
          position: 'fixed',
          bottom: '20px',
          left: '20px',
        }}
      />
      <div className="blog-content">
        <Pane padding={majorScale(2)}>
          {blogPosts.map((post, index) => (
            <Card
              key={index}
              backgroundColor={darkMode ? 'gray' : 'white'}
              elevation={1}
              hoverElevation={2}
              marginY={majorScale(1)}
              padding={majorScale(2)}
              display="flex"
              flexDirection="column"
              justifyContent="space-between"
              onClick={() => handleCardClick(post.link)}
              cursor="pointer"
            >
              <Heading
                size={500}
                marginBottom={majorScale(1)}
                color={darkMode ? '#E4E7EB' : '#172b41'}
              >
                {post.title}
              </Heading>
              <Text size={300} color={darkMode ? 'A7B6C2' : '#5E6C84'}>
                {post.date}
              </Text>
            </Card>
          ))}
        </Pane>
      </div>
    </div>
  );
}

export default Blog;
