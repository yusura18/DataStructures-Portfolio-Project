# Course: CS261 - Data Structures
# Author: Breanna Moore
# Assignment: 6
# Description: This file contains the class DirectedGraph with methods to add, remove,
# and get vertices and edges. The class also contains methods for searching the graph
# checking paths, checking for cycles, and an implementation of Dijkstra's Algorithm
# to find the shortest path from a specified vertex to all other vertices in the graph.


import heapq
from collections import deque


class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        This method adds a new vertex to the graph. Vertex name doesn't
        need to be provided, vertex will be assigned reference index.
        This method returns the number of vertices in the graph after
        addition.
        """
        # Create new vertex and add to graph matrix
        self.adj_matrix.append([0])

        # Update other vertices in graph
        for vertex in range(self.v_count):
            self.adj_matrix[self.v_count].append(0)
            self.adj_matrix[vertex].append(0)

        # Increment vertex count
        self.v_count += 1

        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        This method adds a new edge to the graph, connecting two
        vertices with provided indices. If either (or both) vertices
        don't exist in the graph, or if the weight is not a positive
        integer, or if src and dst refer to the same vertex, method
        does nothing. If edge already exists, method updates that
        edges weight.
        """
        # Check for valid indices and weight
        if src == dst or weight < 0:
            return

        if src > self.v_count - 1 or dst > self.v_count - 1:
            return

        if src < 0 or dst < 0:
            return

        # Create edge or update weight if edge exists
        self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        This method removes an edge between two vertices with given
        indices. If either (or both) vertex indices don't exist,
        or if there is no edge between them, the method does
        nothing.
        """
        # Check if vertex indices are valid
        if src == dst:
            return

        if src > self.v_count - 1 or dst > self.v_count - 1:
            return

        if src < 0 or dst < 0:
            return

        # Remove edge between vertices
        self.adj_matrix[src][dst] = 0

    def get_vertices(self) -> []:
        """
        This method returns a list of vertices of the graph. Order
        of the vertices in the list does not matter.
        """
        vertices = []

        # Add index value in range of v_count to vertices list
        for x in range(self.v_count):
            vertices.append(x)

        return vertices

    def get_edges(self) -> []:
        """
        This method returns a list of edges in the graph. Each edge
        is returned as a tuple of two incident vertex indices and
        weight. The tuple format is (src, dst, weight).
        """
        edges = []

        # Check if graph is empty
        if self.v_count == 0:
            return edges

        # Loop through each vertex of the graph and check the indices
        # of each vertex for an edge.
        for vertex in range(self.v_count):
            for dst_v in range(self.v_count):
                if self.adj_matrix[vertex][dst_v] > 0:
                    # There is an edge present, add tuple to edges list
                    edges.append((vertex, dst_v, self.adj_matrix[vertex][dst_v]))

        return edges

    def is_valid_path(self, path: []) -> bool:
        """
        This method takes a list of vertex indices and returns True
        if the sequence of vertices represents a valid path in the
        graph. An empty path is considered valid.
        """
        # Check if path is empty
        if len(path) == 0:
            return True

        # If one vertex in path
        if len(path) == 1:
            if path[0] < self.v_count:
                return True
            else:
                return False

        # Loop through path list and check if there is an edge
        # in the graph from vertex v to the next vertex of path.
        for v in range(len(path)-1):
            # If value of next vertex is 0, there isn't an edge
            if self.adj_matrix[path[v]][path[v+1]] == 0:
                return False

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        This method performs a DFS in the graph and returns a
        list of vertices visited during the search, in the order
        visited. If starting vertex is not in graph, method
        returns empty list. If end vertex is included as a
        parameter, but is not in graph or can't be reached, a
        list of a complete DFS is returned for the starting
        vertex.
        """
        # Check if starting vertex is in graph
        if v_start > self.v_count - 1:
            return []

        # Create empty visited list and stack
        visited = []
        stack = deque()

        # Add starting vertex to queue
        stack.append(v_start)

        while len(stack) != 0:
            # Remove vertex
            vertex = stack.pop()

            # Add vertex to visited list
            if vertex not in visited:
                visited.append(vertex)

                connections = []
                # Check for any connections in graph at vertex
                for x in range(len(self.adj_matrix[vertex])):
                    # If there is an edge towards vertex x, push it
                    # on heap
                    if self.adj_matrix[vertex][x] != 0:
                        connections.append(x)

                for vertex in reversed(sorted(connections)):
                    if vertex not in visited:
                        stack.append(vertex)


            # Check if reached end vertex
            if vertex == v_end:
                return visited

        return visited

    def bfs(self, v_start, v_end=None) -> []:
        """
        This method performs a BFS in the graph and returns a
        list of vertices visited during the search, in the order
        visited. If starting vertex is not in graph, method
        returns empty list. If end vertex is included as a
        parameter, but is not in graph or can't be reached, a
        list of a complete BFS is returned for the starting
        vertex.
        """
        # Check if starting vertex is in graph
        if v_start > self.v_count - 1:
            return []

        # Create empty visited list and deque
        visited = []

        d = deque()
        d.append(v_start)       # Add starting vertex to deque

        while len(d) != 0:
            # Remove the leftmost vertex from deque
            vertex = d.popleft()

            # Add vertex to visited list
            if vertex not in visited:
                visited.append(vertex)

                # Check for any connections in graph at vertex
                for x in range(len(self.adj_matrix[vertex])):
                    # If there is an edge towards vertex x, append it
                    # to the end of deque
                    if self.adj_matrix[vertex][x] != 0:
                        d.append(x)

            # Check if reached end vertex
            if vertex == v_end:
                return visited

        return visited

    def has_cycle(self):
        """
        This method returns True if there is at least one
        cycle in the graph. If the graph is acyclic,
        returns False.
        """
        # Lists to track visited vertices, the stack with
        # a vertex's connections
        visited = []
        stack = []

        for vertex in range(self.v_count):
            if vertex not in visited:
                if self._has_cycle_helper(vertex, visited, stack):
                    return True

        return False

    def _has_cycle_helper(self, vertex, visited, stack):
        """
        This is a recursive helper method for the has cycle
        method.
        """
        visited.append(vertex)
        stack.append(vertex)

        index = 0
        # Check vertex for edges
        for adj_v in self.adj_matrix[vertex]:
            # Base case
            if index in stack and adj_v != 0:
                return True
            elif index not in visited and adj_v != 0:
                # Recursive call
                if self._has_cycle_helper(index, visited, stack):
                    return True
            index += 1

        stack.remove(vertex)
        return False

    def dijkstra(self, src: int) -> []:
        """
        This method implements Dijkstra's algorithm to compute
        the length of the shortest path from a given vertex to
        all other vertices. Method returns a list with one value
        per each vertex, where the value at index 0 is the length
        of the shortest path from vertex src to vertex 0. If a
        vertex is not reachable from src, returned valued is
        infinity.
        """
        # Initialize empty list for tracking the
        # minimum distance to each vertex from src.
        min_dist = []

        # Check if src is in graph
        if src > self.v_count - 1:
            return []

        # Append min_dist list with infinity for every
        # vertex in graph
        for v in range(self.v_count):
            min_dist.append(float('inf'))

        # Set the src vertex distance to 0
        min_dist[src] = 0

        # Create priority queue with list containing minimum distance
        # and starting vertex src.
        queue = [[0, src]]

        # Loop through priority queue until empty.
        while len(queue) > 0:
            # Pop the vertex with the shortest distance
            current = heapq.heappop(queue)
            cur_dist = current[0]
            vertex = current[1]

            if cur_dist > min_dist[vertex]:
                continue

            # Look for connections at the vertex. If there is a
            # connection, add the weight of the connection to
            # cur_dist.
            for index in range(len(self.adj_matrix[vertex])):
                if self.adj_matrix[vertex][index] != 0:
                    adj_v = index
                    distance = cur_dist + self.adj_matrix[vertex][index]

                    # If the new calculated distance is lower than
                    # distance stored in min_dist list, update that
                    # distance for the adjacent vertex
                    if distance < min_dist[adj_v]:
                        min_dist[adj_v] = distance

                        # Add distance and adjacent vertex to priority queue
                        heapq.heappush(queue, [distance, adj_v])

        return min_dist


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)


    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        g.add_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)


    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
