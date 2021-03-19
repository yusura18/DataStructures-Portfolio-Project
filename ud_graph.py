# Course: CS261 - Data Structures
# Author: Breanna Moore
# Assignment: 6
# Description: This file contains the class UndirectedGraph with methods to add, remove,
# and get vertices and edges. The class also contains methods for searching the graph
# checking paths, and checking for cycles.


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        This method adds a new vertex to the graph. Vertex names can be
        any string. If vertex with name already present, method does
        nothing.
        """
        # Check if dictionary is empty
        if len(self.adj_list) == 0:
            self.adj_list[v] = []

        # Check if dictionary already contains the vertex
        if v in self.adj_list.keys():
            return
        else:
            self.adj_list[v] = []

    def add_edge(self, u: str, v: str) -> None:
        """
        This method adds a new edge to the graph, connecting two vertices
        with provided names. If either (or both) vertex names don't
        exist, this method will create the vertices and then create an
        edge between them. If an edge already exists or if u and v
        refer to the same vertex, method does nothing.
        """
        # Check if vertices are the same
        if u == v:
            return

        # Check if both vertices are in dictionary
        if u in self.adj_list.keys() and v in self.adj_list.keys():
            if v in self.adj_list[u]:
                return
            else:
                self.adj_list.get(u).append(v)
                self.adj_list.get(v).append(u)
                return

        # If vertex not present, add the vertex to dict
        if u not in self.adj_list.keys():
            self.add_vertex(u)
        if v not in self.adj_list.keys():
            self.add_vertex(v)

        # Create the edge
        self.adj_list.get(u).append(v)
        self.adj_list.get(v).append(u)

    def remove_edge(self, v: str, u: str) -> None:
        """
        This method removes an edge between two vertices with provided
        names. If either (or both) vertex names do not exist, or if there
        is no edge between them, the method does nothing.
        """
        # Check if vertices exist
        if v not in self.adj_list.keys() or u not in self.adj_list.keys():
            return

        # Check if edge exists between v and u
        if v in self.adj_list.get(u):
            self.adj_list.get(u).remove(v)
            self.adj_list.get(v).remove(u)
        else:
            # Edge doesn't exist
            return

    def remove_vertex(self, v: str) -> None:
        """
        This method removes a vertex with a given name and all edges
        incident to it from the graph. If given vertex doesn't exist,
        this method does nothing.
        """
        # Check if vertex in dictionary
        if v in self.adj_list.keys():
            for edge in self.adj_list.get(v):
                self.adj_list.get(edge).remove(v)
            del self.adj_list[v]
        else:
            return

    def get_vertices(self) -> []:
        """
        This method returns a list of vertices of the graph in
        any order.
        """
        # For key in dictionary, append key to vertex list
        v_list = []
        for key in self.adj_list:
            v_list.append(key)

        return v_list

    def get_edges(self) -> []:
        """
        This method returns a list of edges in the graph. Each edge
        is returned as a tuple of two incident vertex names.
        """
        edge_list = []

        # Check each key-value pair of dictionary
        for key in self.adj_list:
            for x in self.adj_list[key]:
                if tuple(sorted((key, x))) not in edge_list:
                    edge_list.append(tuple(sorted((key, x))))

        return edge_list

    def is_valid_path(self, path: []) -> bool:
        """
        This method takes a list of vertex names and returns True
        if the sequence of vertices represents a valid path in the
        graph. An empty path is considered valid.
        """
        # Check if path is empty
        if len(path) == 0:
            return True

        # path contains one vertex
        if len(path) == 1:
            if path[0] in self.adj_list.keys():
                return True
            else:
                return False

        # Loop through path list and check if next vertex is
        # in the current vertex's edge list.
        for v in range(len(path)-1):
            if path[v+1] not in self.adj_list[path[v]]:
                return False

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        This method performs a DFS in the graph and returns a
        list of vertices visited during the search in order
        visited. If starting vertex is not in the graph,
        returns an empty list. If end vertex is included as a
        parameter, but is not in graph or can't be reached, a
        list of a complete BFS is returned for the starting
        vertex.
        """
        # Initialize empty lists for tracking vertices
        visited = []
        stack = []

        # Check if starting vertex in dict
        if v_start not in self.adj_list.keys():
            return visited

        stack.append(v_start)

        # Loop until stack is empty
        while len(stack) != 0:
            # Pop vertex from stack
            vertex = stack.pop()

            # Add vertex to visited if not already in list
            if vertex not in visited:
                visited.append(vertex)

                # Loop through the vertices with connections to vertex in
                # descending order and add them to stack
                for x in reversed(sorted(self.adj_list[vertex])):
                    stack.append(x)

            # Check if current vertex matches end vertex
            if vertex == v_end:
                return visited

        return visited

    def bfs(self, v_start, v_end=None) -> []:
        """
        This method performs a BFS in the graph and returns a
        list of vertices visited during the search in the order
        visited. If starting vertex not in graph, returns empty
        list. If end vertex is included as a parameter, but is
        not in graph or can't be reached, a list of a complete
        BFS is returned for the starting vertex.
        """
        # Initialize empty lists for tracking vertices
        visited = []
        queue = []

        # Check if starting vertex in dict and append it to queue
        if v_start not in self.adj_list.keys():
            return visited

        queue.append(v_start)

        # Loop until queue is empty
        while len(queue) != 0:
            # Pop the vertex at index 0
            vertex = queue.pop(0)

            # Add vertex to visited if not already in list
            if vertex not in visited:
                visited.append(vertex)

                # Loop through the vertices with connections to vertex in
                # ascending order
                for x in sorted(self.adj_list[vertex]):
                    # If the vertex "x" has not already been visited,
                    # add to queue
                    if x not in visited:
                        queue.append(x)

            # If the current vertex matches the end vertex,
            # return visited list.
            if vertex == v_end:
                return visited

        return visited

    def count_connected_components(self):
        """
        This method returns the number of connected components
        in the graph.
        """
        # Get list of vertices and initialize count variable
        v_unconnected = self.get_vertices()
        count = 0

        while len(v_unconnected) != 0:
            vertex = v_unconnected.pop()
            # Perform DFS on vertex to get a list of connected
            # components.
            vertex_dfs = self.dfs(vertex)

            # Search connections from vertex_dfs. If a corresponding
            # connection vertex is in unconnected list, remove it.
            for v in vertex_dfs:
                if v in v_unconnected:
                    v_unconnected.remove(v)

            count += 1              # Increment count

        return count

    def has_cycle(self):
        """
        This method returns True if there is at least one cycle
        in the graph. If the graph is acyclic, method returns
        False.
        """
        visited = []
        stack = []
        all_vertices = self.get_vertices()

        # Loop through vertex list, checking its connections in
        # depth first search
        for vertex in all_vertices:
            stack.append(vertex)

            while len(stack) != 0:
                cur_vertex = stack.pop()

                # Add vertex to visited if not already in list
                if cur_vertex not in visited:
                    visited.append(cur_vertex)

                    # Loop through vertex connections in descending order
                    # and append them to stack
                    for x in reversed(sorted(self.adj_list[cur_vertex])):
                        stack.append(x)

                # If cur_vertex is in stack, the graph contains a cycle
                if cur_vertex in stack:
                    return True

            # Empty visited and stack to check next vertex
            visited = []
            stack = []

        return False        # No cycles found


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)


    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)


    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')


    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())
