import '../../styles/blogPost.css';

import { Card, Heading, majorScale, Pane, Text } from 'evergreen-ui';
import { useNavigate } from 'react-router-dom';

function Blog() {
  const navigate = useNavigate();

  const blogPosts = [
    {
      title: `Computer Systems Organization 2`,
      date: 'May 1, 2024',
      link: `/notes/cso2`,
    },
  ];

  const handleCardClick = (path: string) => {
    navigate(path);
  };

  return (
    <div className="blog-content">
      <Pane padding={majorScale(2)}>
        {blogPosts.map((post, index) => (
          <Card
            key={index}
            className="BlogCard"
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
            <Heading size={500} marginBottom={majorScale(1)} className="blogPostHeading">
              {post.title}
            </Heading>
            <Text size={300} color={'A7B6C2'}>
              {post.date}
            </Text>
          </Card>
        ))}
      </Pane>
    </div>
  );
}

export default Blog;
