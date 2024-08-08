# Example: Algorithms

<span class="subtitle">
    Spring 2023
</span>

## Graphs

A graph is a mathematical structure that represents relationships between objects. It consists of vertices (nodes) and edges (connections between vertices).

<blockquote class="definition">

A **graph** $G$ is a pair $(V, E)$, where:

- $V$ is a set of vertices or nodes.
- $E$ is a set of edges connecting the vertices.

</blockquote>

We can calculate the number of edges in a graph using the formula:

<blockquote class="equation">

$$
|E| = \frac{1}{2} \sum_{v \in V} \text{deg}(v),
$$

where $\text{deg}(v)$ is the degree of vertex $v$, i.e., the number of edges incident to $v$.

</blockquote>

Suppose we are interested in the following graph for the following discussion:

```tikz
\usepackage{tikz}
\begin{document}

\begin{tikzpicture}[every node/.style={minimum size=1cm}, scale=2.5]
  % Nodes
  \node[shape=circle,draw=black] (A) at (0,0) {A};
  \node[shape=circle,draw=black] (B) at (2,0) {B};
  \node[shape=circle,draw=black] (C) at (2,2) {C};
  \node[shape=circle,draw=black] (D) at (0,2) {D};
  \node[shape=circle,draw=black] (E) at (3,-1) {E};
  \node[shape=circle,draw=black] (F) at (-1,-1) {F};

  % Edges
  \draw (A) -- (B) node[midway, above] {2};
  \draw (B) -- (C) node[midway, left] {3};
  \draw (C) -- (D) node[midway, above] {4};
  \draw (D) -- (A) node[midway, right] {1};
  \draw (B) -- (E) node[midway, above] {5};
  \draw (A) -- (F) node[midway, below] {6};
  \draw (F) -- (D) node[midway, left] {7};
  \draw (E) -- (C) node[midway, right] {8};
  \draw (A) -- (E) node[midway, below] {9};
  \draw (B) -- (D) node[midway, above] {10};
  \draw (E) -- (F) node[midway, below] {11};
\end{tikzpicture}

\end{document}
```

We can represent graphs using an adjacency matrix or an adjacency list.

### Adjacency Matrix

An adjacency matrix is a 2D array where the entry at row $i$ and column $j$ represents the presence of an edge between vertices $i$ and $j$.

$$
\begin{bmatrix}
0 & 2 & 0 & 1 & 9 & 6 \\
2 & 0 & 3 & 10 & 5 & 0 \\
0 & 3 & 0 & 4 & 8 & 0 \\
1 & 10 & 4 & 0 & 0 & 7 \\
9 & 5 & 8 & 0 & 0 & 11 \\
6 & 0 & 0 & 7 & 11 & 0
\end{bmatrix}
$$

### Adjacency List

An adjacency list is a collection of lists or arrays where each list represents the neighbors of a vertex.

$$
\begin{align*}
A & : \{B, D, F, E\} \\
B & : \{A, C, E, D\} \\
C & : \{B, D, E\} \\
D & : \{A, B, C, F\} \\
E & : \{A, B, C, F\} \\
F & : \{A, D, E\}
\end{align*}
$$

<blockquote class="important">

**Adjacency Matrix** has a space complexity of $O(V^2)$, where $V$ is the number of vertices.

**Adjacency List** has a space complexity of $O(V + E)$, where $E$ is the number of edges.

</blockquote>

Let's implement the adjacency matrix and adjacency list for the graph above.

```python
# Adjacency Matrix
G = [
    [0, 2, 0, 1, 9, 6],
    [2, 0, 3, 10, 5, 0],
    [0, 3, 0, 4, 8, 0],
    [1, 10, 4, 0, 0, 7],
    [9, 5, 8, 0, 0, 11],
    [6, 0, 0, 7, 11, 0]
]
```

## Shortest Path

The shortest path problem involves finding the shortest path between two vertices in a graph. There are several algorithms to solve this problem, such as Dijkstra's algorithm and Bellman-Ford algorithm.

### Bellman-Ford Algorithm

The Bellman-Ford algorithm is used to find the shortest path from a source vertex to all other vertices in a weighted graph. It can handle negative edge weights but detects negative cycles.

```execute-python
def bellman_ford(graph, source):
    # Initialize distances
    distances = {vertex: float('inf') for vertex in graph}
    distances[source] = 0

    # Relax edges repeatedly
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u]:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight

    # Check for negative cycles
    for u in graph:
        for v, weight in graph[u]:
            if distances[u] + weight < distances[v]:
                return "Graph contains negative cycle"

    return distances

# Graph represented as adjacency list
graph = {
    'A': [('B', 2), ('D', 1), ('F', 6), ('E', 9)],
    'B': [('A', 2), ('C', 3), ('E', 5), ('D', 10)],
    'C': [('B', 3), ('D', 4), ('E', 8)],
    'D': [('A', 1), ('B', 10), ('C', 4), ('F', 7)],
    'E': [('A', 9), ('B', 5), ('C', 8), ('F', 11)],
    'F': [('A', 6), ('D', 7), ('E', 11)]
}

source = 'A'
print(bellman_ford(graph, source))
```

### Dijkstra's Algorithm

Dijkstra's algorithm is used to find the shortest path from a source vertex to all other vertices in a weighted graph. It is more efficient than the Bellman-Ford algorithm for non-negative edge weights.

```execute-python
import heapq

def dijkstra(graph, source):
    distances = {vertex: float('inf') for vertex in graph}
    distances[source] = 0
    pq = [(0, source)]

    while pq:
        dist_u, u = heapq.heappop(pq)
        if dist_u > distances[u]:
            continue

        for v, weight in graph[u]:
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                heapq.heappush(pq, (distances[v], v))

    return distances

graph = {
    'A': [('B', 2), ('D', 1), ('F', 6), ('E', 9)],
    'B': [('A', 2), ('C', 3), ('E', 5), ('D', 10)],
    'C': [('B', 3), ('D', 4), ('E', 8)],
    'D': [('A', 1), ('B', 10), ('C', 4), ('F', 7)],
    'E': [('A', 9), ('B', 5), ('C', 8), ('F', 11)],
    'F': [('A', 6), ('D', 7), ('E', 11)]
}

source = 'A'
print(dijkstra(graph, source))
```

## Conclusion

Algorithms are cool.
