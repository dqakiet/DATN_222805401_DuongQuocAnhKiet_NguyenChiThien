import itertools
import heapq
import math

class Graph:
    def __init__(self) -> None:
        """
        Initializes an instance of the Graph class.
        """
        self.adj_list: dict[str, list[tuple[str, float]]] = {}

    def add_vertex(self, vertex: str) -> None:
        """
        Adds a vertex to the graph.
        :param vertex: The name of the vertex to be added.
        """
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, u: str, v: str, w: float, p: float) -> None:
        """
        Adds an edge to the graph with a specified probability.
        :param u: The name of the first vertex.
        :param v: The name of the second vertex.
        :param w : The weight of the edge connecting vertex u and v
        :param p: The probability of the edge connecting vertex u and v.
        """
        if u not in self.adj_list:
            self.add_vertex(u)
        if v not in self.adj_list:
            self.add_vertex(v)
        self.adj_list[u].append((v, w, p))
        self.adj_list[v].append((u, w, p))

    def remove_vertex(self, vertex: str, temp_adj_list: dict[str, list[tuple[str, float]]]) -> None:
        """
        Removes a vertex and its associated edges from a temporary adjacency list.
        :param vertex: The vertex to be removed.
        :param temp_adj_list: A temporary adjacency list from which the vertex is removed.
        """
        for u, _, _ in temp_adj_list[vertex]:
            temp_adj_list[u] = [(v, w, p) for v, w, p in temp_adj_list[u] if v != vertex]
        del temp_adj_list[vertex]

    def get_weighted_expected_degree(self, vertex: str, temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the weighted expected degree of a vertex based on edge probabilities.
        :param vertex: The vertex whose weighted expected degree is to be calculated.
        :param temp_adj_list: The adjacency list used for calculation.
        :return: The weighted expected degree of the given vertex.
        """
        return sum((weight * prob) for _, weight, prob in temp_adj_list[vertex])

    def get_weighted_expected_density(self, vertices: set[str],
                                   temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the weighted expected density of a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The weighted expected density of the subgraph.
        """
        total_prob = sum(
            (weight * prob) for v in vertices for u, weight, prob in temp_adj_list[v] if u in vertices and u > v)
        return total_prob / len(vertices) if vertices else 0

    def get_excess_degree(self, vertex: str, bound: float, temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the excess degree of a vertex.
        :param vertex: The vertex whose excess degree is to be calculated.
        :param bound: The  bound value.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The excess degree of the given vertex.
        """
        return sum((weight * (prob - bound)) for _, weight, prob in temp_adj_list[vertex])

    def get_excess_average_degree(self, vertices: set[str], bound: float,
                                         temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the excess average degree of a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :param bound: The bound value.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The excess average degree of the subgraph.
        """
        total_prob = 0
        num_edges = 0
        for v in vertices:
            for u, weight, prob in temp_adj_list[v]:
                if u in vertices and u > v:
                    total_prob += weight * (prob - bound)
                    num_edges += 1
        return (total_prob) / len(vertices) if num_edges > 0 else 0

    def greedy_bwds(self, bound: float) -> tuple[set[str], float]:
        """
        Applies a greedy algorithm to find the subgraph with the highest excess average degree.
        :param bound: The bound value.
        :return:  A tuple containing the set of vertices forming the best subgraph and its excess average degree.
        """
        temp_adj_list = {v: edges.copy() for v, edges in self.adj_list.items()}
        vertices = set(self.adj_list.keys())
        best_subgraph = set()
        best_excess_avg_degree = 0

        heap = []
        vertex_to_marker = {}
        current_marker = 0

        for v in vertices:
            excess_degree = self.get_excess_degree(v, bound, temp_adj_list)
            heapq.heappush(heap, (excess_degree, v, current_marker))
            vertex_to_marker[v] = current_marker

        while len(vertices) >= 2:
            current_excess_avg_degree = self.get_excess_average_degree(vertices, bound, temp_adj_list)
            if current_excess_avg_degree > best_excess_avg_degree:
                best_excess_avg_degree = current_excess_avg_degree
                best_subgraph = set(vertices)

            while heap:
                excess_degree, v_to_remove, marker = heapq.heappop(heap)
                if v_to_remove in vertices and vertex_to_marker[v_to_remove] == marker:
                    vertices.remove(v_to_remove)
                    break

            neighbors = [neighbor for neighbor, _, _ in temp_adj_list[v_to_remove]]
            self.remove_vertex(v_to_remove, temp_adj_list)

            current_marker += 1
            for neighbor in neighbors:
                if neighbor in vertices:
                    new_degree = self.get_excess_degree(neighbor, bound, temp_adj_list)
                    heapq.heappush(heap, (new_degree, neighbor, current_marker))
                    vertex_to_marker[neighbor] = current_marker

        return best_subgraph, round(best_excess_avg_degree, 3)

    def greedy_uwds(self) -> tuple[set[str], float]:
        """
        Applies a greedy algorithm to find the subgraph with the highest expected edge density.
        :return: A tuple containing the set of vertices forming the best subgraph and its weighted expected density.
        """
        temp_adj_list = {v: edges.copy() for v, edges in self.adj_list.items()}
        vertices = list(temp_adj_list.keys())
        best_subgraph = set()
        best_density = 0

        while len(vertices) >= 2:
            current_density = self.get_weighted_expected_density(set(vertices), temp_adj_list)
            if current_density > best_density:
                best_density = current_density
                best_subgraph = set(vertices)
            v = min(vertices, key=lambda x: self.get_weighted_expected_degree(x, temp_adj_list))
            self.remove_vertex(v, temp_adj_list)
            vertices.remove(v)

        return best_subgraph, round(best_density, 3)

    def brute_force_search(self) -> tuple[set[str], float]:
        """
        Applies a brute force algorithm to find the subgraph with the highest weighted expected density.
        :return: A tuple containing the set of vertices forming the best subgraph and its weighted expected density
        """
        best_subgraph = set()
        best_weighted_expected_density = 0
        vertices = list(self.adj_list.keys())

        edges = set()
        for vertex in vertices:
            for neighbor, _, _ in self.adj_list[vertex]:
                if (vertex, neighbor) not in edges and (neighbor, vertex) not in edges:
                    edges.add((vertex, neighbor))
        for r in range(1, len(edges) + 1):
            for edge_subset in itertools.combinations(edges, r):
                subgraph_vertices = set()
                for u, v in edge_subset:
                    subgraph_vertices.add(u)
                    subgraph_vertices.add(v)
                subgraph_set = set(subgraph_vertices)
                weighted_expected_density = self.get_weighted_expected_density(subgraph_set, self.adj_list)
                if weighted_expected_density > best_weighted_expected_density:
                    best_weighted_expected_density = weighted_expected_density
                    best_subgraph = subgraph_set

        return best_subgraph, best_weighted_expected_density

    def adjoint_logarithmic_reliability(self, vertices: set[str]) -> float:
        """
        Calculates the adjoint reliability of a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :return: The adjoint reliability of the subgraph.
        """
        reliability = 0
        for v in vertices:
            for u, weight, prob in self.adj_list[v]:
                if u in vertices and u > v:
                    reliability += math.log10(prob)
        return reliability

    def average_edge_weighted_probability(self, vertices: set[str]) -> float:
        """
        Calculates the average edge probability of a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :return: The average edge probability of the subgraph.
        """
        total_prob = 0
        num_edges = 0
        for v in vertices:
            for u, weight, prob in self.adj_list[v]:
                if u in vertices and u > v:
                    total_prob += prob * weight
                    num_edges += 1
        return total_prob / num_edges if num_edges > 0 else 0

    def weighted_expected_edge_density(self, vertices: set[str]) -> float:
        """
        Calculates the expected edge density of a subgraph.
        :param vertices:  A set of vertices forming the subgraph.
        :return: The expected edge density of the subgraph.
        """
        if len(vertices) < 2:
            return 0
        total_prob = sum(
            (weight * prob) for v in vertices for u, weight, prob in self.adj_list[v] if u in vertices and u > v)
        max_edges = len(vertices) * (len(vertices) - 1) / 2
        return total_prob / max_edges

    def std_egde_weigh_probability(self, vertices: set[str]) -> float:
        """
        Calculates the weighted standard deviation of edge probabilities in a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :return: The weighted standard deviation of edge probabilities in the subgraph.
        """
        edge_values = []
        weights = []
        total_weight = 0

        for v in vertices:
            for u, weight, prob in self.adj_list[v]:
                if u in vertices and u > v:
                    edge_prob = weight * prob
                    edge_values.append(edge_prob)
                    weights.append(weight)
                    total_weight += weight
        if not edge_values:
            return 0
        weighted_average = sum(weights[i] * edge_values[i] for i in range(len(edge_values))) / total_weight
        weighted_variance = sum(
            weights[i] * ((edge_values[i] - weighted_average) ** 2) for i in range(len(edge_values))) / total_weight

        return math.sqrt(weighted_variance)

    def evaluation_metric(self, subgraph: set[str]) -> tuple[int, int, dict[str, float]]:
        """
        Evaluates a subgraph using various metrics.
        :param subgraph: A set of vertices forming the subgraph.
        :return: A tuple of the number of vertices, the number of edges, and a dictionary of various evaluation metrics for the subgraph.
        """
        eed = self.weighted_expected_edge_density(subgraph)
        aep = self.average_edge_weighted_probability(subgraph)
        eps = self.std_egde_weigh_probability(subgraph)
        ar = self.adjoint_logarithmic_reliability(subgraph)
        vertices = len(subgraph)

        num_edges = len(
            {(u, v) for v in subgraph for u, _, _ in self.adj_list[v] if u in subgraph and v in subgraph and u > v})
        # print("Subgraph vertices (Total: {}):".format(len(subgraph)), subgraph)
        # num_edges = len(
        #     {(u, v) for v in subgraph for u, _, _ in self.adj_list[v] if u in subgraph and v in subgraph and u > v})
        subgraph_dict = {vertex: [] for vertex in subgraph}
        for vertex in subgraph:
            for neighbor, weight, prob in self.adj_list[vertex]:
                if neighbor in subgraph:
                    subgraph_dict[vertex].append((neighbor, weight, prob))
        #
        # print("Subgraph vertices (Total: {}):".format(len(subgraph)))
        # print(len(subgraph_dict))
        metrics_dict = {
            'Weighted Expected Edge Density': round(eed, 3),
            'Average Edge Weighted Probability': round(aep, 3),
            'The Standard Deviation of Edge Weight Probability': round(eps, 3),
            'Adjoint Logarithmic Reliability': round(ar, 3),
        }
        return vertices, subgraph_dict, num_edges, metrics_dict

    def print_summarize_graph(self):
        """
        Prints details of a graph including its vertices and edges.
        """
        num_edges = 0
        vertices = set(self.adj_list.keys())
        print("Graph vertices (Total: {}):".format(len(vertices)))
        for v in vertices:
            for u, _, _ in self.adj_list[v]:
                if u in vertices and u > v:
                    num_edges += 1
        print("Num Edges:", num_edges)
        print("Average edge probability weighted :", round(self.average_edge_weighted_probability(vertices), 3))
        print("edge weighted probability std :", round(self.std_egde_weigh_probability(vertices), 3))
