import '../styles/CourseTree.css';

import React, { useEffect, useRef, useState } from 'react';
import Tree from 'react-d3-tree';

const orgChart = {
  name: 'UVA',
  children: [
    {
      name: 'Introduction to Programming',
      attributes: {
        Achievements: 'Created a spaceship dropdown game using PyGame.',
        Language: 'Python',
      },
      children: [
        {
          name: 'Computer Systems & Organization',
          attributes: {
            Achievement: "Lab Lead Teaching Assistant for Spring '23",
            Language: 'C, x86 Assembly',
          },
          children: [
            {
              name: 'Computer Systems & Architecture',
              attributes: {
                Status: 'In Progress',
              },
            },
          ],
        },
        {
          name: 'Software Development Essentials',
          attributes: {
            Achievement: 'Created Full-Stack Course Review App in Java.',
            Language: 'Java, SQL, Gradle, Mockito, JavaFX, Junit5',
          },
          children: [
            {
              name: 'Advanced Software Development',
              attributes: {
                Achievement: 'Created Full-Stack AI Map Assistant in Django.',
                Tools:
                  'Django, Python, HTML, CSS, JavaScript, Bootstrap, Heroku, PostgreSQL, Git',
              },
            },
          ],
        },
        {
          name: 'Discrete Mathematics',
          attributes: {
            Language: 'Lean Language',
          },
          children: [
            {
              name: 'Theory of Computation',
              attributes: {
                Language: 'LaTeX, Java',
                Achievement: 'Created a NFA simulator in Java.',
              },
            },
          ],
        },
        {
          name: 'Data Structures',
          attributes: {
            Language: 'Java',
          },
          children: [
            {
              name: 'Algorithms',
              attributes: {
                Language: 'Python, Java, LaTeX',
              },
              children: [
                {
                  name: 'Optimization',
                  attributes: {
                    Achievement: 'Implemented NN from Scratch.',
                    Algorithms:
                      "Newton's Method, Projected GD, Mirror Descent, Proximal GD",
                    Math: 'KKT, Conjugate Functions, L-Smoothness, Strong Convexity',
                    Language: 'Python, Numpy',
                  },
                },
                {
                  name: 'Machine Learning',
                  attributes: {
                    Description:
                      'Linear + Logistic Regression, K-Means, SVMs, Bayesian Learning, ANN, CNN, RNN + LSTM, LLM, VAE',
                    Language: 'Python, PyTorch, TensorFlow, Pandas, Jupyter',
                  },
                  children: [
                    {
                      name: 'Natural Language Processing',
                      attributes: {
                        Status: 'In Progress',
                      },
                    },
                    {
                      name: 'Probabilistic ML',
                      attributes: {
                        Status: 'In Progress',
                      },
                    },
                    {
                      name: 'Deep Learning',
                      attributes: {
                        Status: 'In Progress',
                      },
                    },
                    {
                      name: 'Reinforcement Learning',
                      attributes: {
                        Achievement:
                          'Created a Multi-Agent RL agent to play Tetris. Achieved SOTA performance in multiplayer Tetris.',
                        Description:
                          'Multi-Armed Bandits, MDPs, DP, Monte-Carlo, TD, Policy Gradient, Approximation Methods, Deep RL, Offline RL',
                        Language: 'Python, PyTorch',
                      },
                    },
                  ],
                },
              ],
            },
          ],
        },
      ],
    },
    {
      name: 'Multivariable Calculus',
      children: [
        {
          name: 'Linear Algebra',
          attributes: {
            Language: 'MATLAB',
          },
          children: [
            {
              name: 'Probability',
              attributes: {
                Achievement: 'Created a Monte Carlo Simulation.',
                Language: 'Python',
              },
            },
          ],
        },
      ],
    },
  ],
};

export default function OrgChartTree() {
  const treeContainerRef = useRef<HTMLDivElement>(null);
  const [translate, setTranslate] = useState({ x: 100, y: 100 });

  useEffect(() => {
    const resizeTree = () => {
      const containerWidth = treeContainerRef.current?.clientWidth;
      const containerHeight = treeContainerRef.current?.clientHeight;

      if (containerWidth && containerHeight) {
        setTranslate({
          x: containerWidth / 2,
          y: containerHeight / 50,
        });
      }
    };

    resizeTree();

    window.addEventListener('resize', resizeTree);

    return () => {
      window.removeEventListener('resize', resizeTree);
    };
  }, []);

  return (
    <div
      id="treeWrapper"
      ref={treeContainerRef}
      style={{ width: '100%', height: '100vh' }}
    >
      <Tree
        // @ts-ignore
        data={orgChart}
        rootNodeClassName="node_root"
        branchNodeClassName="node_branch"
        leafNodeClassName="node_leaf"
        orientation="vertical"
        collapsible={false}
        zoomable={true}
        draggable={true}
        centeringTransitionDuration={1000}
        shouldCollapseNeighborNodes={false}
        translate={translate}
        enableLegacyTransitions={true}
        zoom={1}
        hasInteractiveNodes={true}
        nodeSize={{ x: 200, y: 200 }}
        scaleExtent={{ min: 0.1, max: 5 }}
        separation={{ siblings: 3, nonSiblings: 1.5 }}
      />
    </div>
  );
}
