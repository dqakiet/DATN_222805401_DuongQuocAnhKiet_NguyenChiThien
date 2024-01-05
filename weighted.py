import time
import itertools
import heapq
import random
import math


class Graph:
    def __init__(self) -> None:
        """
        Initializes an instance of the Graph class.
        """
        self.adj_list: dict[str, list[tuple[str, float, float]]] = {}

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

    def get_expected_degree(self, vertex: str, temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the expected degree of a vertex based on edge probabilities.
        :param vertex: The vertex whose expected degree is to be calculated.
        :param temp_adj_list: The adjacency list used for calculation.
        :return: The expected degree of the given vertex.
        """
        return sum((weight * prob) for _, weight, prob in temp_adj_list[vertex])

    def calculate_expected_density(self, vertices: set[str],
                                   temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the expected density of a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The expected density of the subgraph.
        """
        total_prob = sum(
            (weight * prob) for v in vertices for u, weight, prob in temp_adj_list[v] if u in vertices and u > v)
        return total_prob / len(vertices) if vertices else 0

    def get_surplus_degree(self, vertex: str, beta: float, temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the surplus degree of a vertex.
        :param vertex: The vertex whose surplus degree is to be calculated.
        :param beta: The  beta value.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The surplus degree of the given vertex.
        """
        return sum((weight * (prob - beta)) for _, weight, prob in temp_adj_list[vertex])

    def calculate_surplus_average_degree(self, vertices: set[str], beta: float,
                                         temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the surplus average degree of a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :param beta: The beta value.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The surplus average degree of the subgraph.
        """
        total_prob = 0
        num_edges = 0
        for v in vertices:
            for u, weight, prob in temp_adj_list[v]:
                if u in vertices and u > v:
                    total_prob += weight * (prob - beta)
                    num_edges += 1
        return (total_prob) / len(vertices) if num_edges > 0 else 0

    def greedy_obetas(self, beta: float) -> tuple[set[str], float]:
        """
        Applies a greedy algorithm to find the subgraph with the highest surplus average degree.
        :param beta: The beta value.
        :return:  A tuple containing the set of vertices forming the best subgraph and its surplus average degree.
        """
        temp_adj_list = {v: edges.copy() for v, edges in self.adj_list.items()}
        vertices = set(self.adj_list.keys())
        best_subgraph = set()
        best_surplus_avg_degree = 0

        heap = []
        vertex_to_marker = {}
        current_marker = 0

        for v in vertices:
            surplus_degree = self.get_surplus_degree(v, beta, temp_adj_list)
            heapq.heappush(heap, (surplus_degree, v, current_marker))
            vertex_to_marker[v] = current_marker

        while len(vertices) >= 2:
            current_surplus_avg_degree = self.calculate_surplus_average_degree(vertices, beta, temp_adj_list)
            print(current_surplus_avg_degree)
            if current_surplus_avg_degree > best_surplus_avg_degree:
                best_surplus_avg_degree = current_surplus_avg_degree
                best_subgraph = set(vertices)

            while heap:
                surplus_degree, v_to_remove, marker = heapq.heappop(heap)
                if v_to_remove in vertices and vertex_to_marker[v_to_remove] == marker:
                    vertices.remove(v_to_remove)
                    break

            neighbors = [neighbor for neighbor, _, _ in temp_adj_list[v_to_remove]]
            self.remove_vertex(v_to_remove, temp_adj_list)

            current_marker += 1
            for neighbor in neighbors:
                if neighbor in vertices:
                    new_degree = self.get_surplus_degree(neighbor, beta, temp_adj_list)
                    heapq.heappush(heap, (new_degree, neighbor, current_marker))
                    vertex_to_marker[neighbor] = current_marker

        return best_subgraph, round(best_surplus_avg_degree, 3)

    def greedy_uds(self) -> tuple[set[str], float]:
        """
        Applies a greedy algorithm to find the subgraph with the highest expected edge density.
        :return: A tuple containing the set of vertices forming the best subgraph and its expected density.
        """
        temp_adj_list = {v: edges.copy() for v, edges in self.adj_list.items()}
        vertices = list(temp_adj_list.keys())
        best_subgraph = set()
        best_density = 0

        while len(vertices) >= 2:
            current_density = self.calculate_expected_density(set(vertices), temp_adj_list)
            if current_density > best_density:
                best_density = current_density
                best_subgraph = set(vertices)
            v = min(vertices, key=lambda x: self.get_expected_degree(x, temp_adj_list))
            self.remove_vertex(v, temp_adj_list)
            vertices.remove(v)

        return best_subgraph, round(best_density, 3)

    def brute_force_search(self) -> tuple[set[str], float]:
        """
        Applies a brute force algorithm to find the subgraph with the highest expected density.
        :return: A tuple containing the set of vertices forming the best subgraph and its expected density
        """
        best_subgraph = set()
        best_expected_density = 0
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
                expected_density = self.calculate_expected_density(subgraph_set, self.adj_list)
                if expected_density > best_expected_density:
                    best_expected_density = expected_density
                    best_subgraph = subgraph_set

        return best_subgraph, best_expected_density

    def adjoint_reliability(self, vertices: set[str]) -> float:
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

    def expected_edge_density(self, vertices: set[str]) -> float:
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

    def edge_weighted_probability_std(self, vertices: set[str]) -> float:
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
        weighted_variance = sum(weights[i] * ((edge_values[i] - weighted_average) ** 2) for i in range(len(edge_values))) / total_weight

        return math.sqrt(weighted_variance)

    def evaluation_metric(self, subgraph: set[str]) -> tuple[int, int, dict[str, float]]:
        """
        Evaluates a subgraph using various metrics.
        :param subgraph: A set of vertices forming the subgraph.
        :return: A tuple of the number of vertices, the number of edges, and a dictionary of various evaluation metrics for the subgraph.
        """
        eed = self.expected_edge_density(subgraph)
        aep = self.average_edge_weighted_probability(subgraph)
        eps = self.edge_weighted_probability_std(subgraph)
        ar = self.adjoint_reliability(subgraph)
        vertices = len(subgraph)

        num_edges = len(
            {(u, v) for v in subgraph for u, _, _ in self.adj_list[v] if u in subgraph and v in subgraph and u > v})

        metrics_dict = {
            'Expected edge density': round(eed, 3),
            'Average edge weighted probability': round(aep, 3),
            'Edge weighted probability std': round(eps, 3),
            'Adjoint Reliability': round(ar, 3),
        }
        return vertices, num_edges, metrics_dict

    def print_subgraph(self, vertices: set[str]) -> None:
        """
        Prints details of a subgraph including its vertices and edges.
        :param vertices: A set of vertices forming the subgraph.
        """
        print("Subgraph vertices (Total: {}):".format(len(vertices)), vertices)
        num_edges = len(
            {(u, v) for v in vertices for u, _, _ in self.adj_list[v] if u in vertices and v in vertices and u > v})
        print("Num Edges:", num_edges)

    def print_summarize_graph(self):
        """
        Prints details of a graph including its vertices and edges.
        """
        num_edges = 0
        vertices = set(self.adj_list.keys())
        print("Graph vertices (Total: {}):".format(len(vertices)), vertices)
        for v in vertices:
            for u, _, _ in self.adj_list[v]:
                if u in vertices and u > v:
                    num_edges += 1
        print("Num Edges:", num_edges)
        print("Average edge probability weighted :", self.average_edge_weighted_probability(vertices))
        printed_edges = set()
        for vertex in vertices:
            for neighbor, weight, prob in self.adj_list[vertex]:
                if neighbor in vertices and (neighbor, vertex) not in printed_edges:
                    print(f"({vertex}, {neighbor}): {weight} {prob}")
                    printed_edges.add((vertex, neighbor))


# test random graph
def create_random_graph(num_vertices):
    """
    Create a random graph
    :param num_vertices: Number vertices of graph
    :return: Object graph with random edges
    """
    g = Graph()

    for i in range(num_vertices - 1):
        weight = random.randint(1, 100)
        probability = random.random()
        g.add_edge(chr(65 + i), chr(65 + i + 1), weight, probability)

    for i in range(num_vertices):
        for j in range(i + 2, num_vertices):
            if random.random() < 0.5:
                weight = random.randint(1, 100)
                probability = random.random()
                g.add_edge(chr(65 + i), chr(65 + j), weight, probability)
    return g


# g = create_random_graph(6)
# g.print_summarize_graph()

g = Graph()
g.add_edge(1, 2, 400, 0.7)
g.add_edge(1, 3, 600, 0.6)
g.add_edge(1, 5, 300, 0.5)
g.add_edge(1, 6, 600, 0.3)
g.add_edge(2, 3, 200, 0.4)
g.add_edge(2, 4, 300, 0.5)
g.add_edge(3, 4, 100, 0.6)
g.add_edge(4, 5, 500, 0.4)
g.add_edge(6, 7, 700, 0.8)
g.add_edge(6, 8, 800, 0.9)
g.add_edge(7, 8, 900, 0.9)
start_time = time.perf_counter()
best_subgraph_uds, best_density_uds = g.greedy_uds()
end_time = time.perf_counter()
execution_time_uds = end_time - start_time
print("GreedyUDS time: ", execution_time_uds)
print("Best subgraph from GreedyUDS:")
g.print_subgraph(best_subgraph_uds)
print("With expected density:", best_density_uds)
vertices_uds, num_edge_uds, eval_uds = g.evaluation_metric(best_subgraph_uds)
print("Evaluation metric: ", eval_uds)
# beta = 0.7
# start_time = time.perf_counter()
# best_subgraph_obetas, best_surplus_avg_degree_obetas = g.greedy_obetas(beta)
# end_time = time.perf_counter()
# execution_time_obetas = end_time - start_time
# print("GreedyUβS time: ", execution_time_obetas)
# print("Best subgraph from GreedyObetaS:")
# g.print_subgraph(best_subgraph_obetas)
# print("With surplus average degree:", best_surplus_avg_degree_obetas)
# vertices_obetas, num_edge_obetas, eval_obetas = g.evaluation_metric(best_subgraph_obetas)
# print("Evaluation metric: ", eval_obetas)
# g = Graph()
#
#
# def read_file(n):
#     u = []
#     v = []
#     w = []
#     p = []
#     with open('579138.protein.links.detailed.v12.0.txt', 'r') as f:
#         lines = f.readlines()
#         header = lines[0].split()
#         for line in lines[1:]:
#             parts = line.split()
#             u.append(parts[0])
#             v.append(parts[1])
#             w.append(float(parts[n]))
#             p.append(parts[-1])
#     return u, v, w, p
#
#
# for i in range(2, 9, 1):
#     g = Graph()
#     print("*******************************************************")
#     print(i)
#     u, v, w, p = read_file(i)
#     for j in range(len(u)):
#         if w[j] == 0.0:
#             w[j] = 0.041 * 1000
#         g.add_edge(u[j], v[j], w[j], int(p[j]) / 1000)
#
#     start_time = time.perf_counter()
#     best_subgraph_uds, best_density_uds = g.greedy_uds()
#     end_time = time.perf_counter()
#     execution_time_uds = end_time - start_time
#     print("GreedyUDS time: ", execution_time_uds)
#     print("Best subgraph from GreedyUDS:")
#     g.print_subgraph(best_subgraph_uds)
#     print("With expected density:", best_density_uds)
#     vertices_uds, num_edge_uds, eval_uds = g.evaluation_metric(best_subgraph_uds)
#     print("Evaluation metric: ", eval_uds)
#
#     print("--------------------------------------------------------")
#     betas = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
#     for beta in betas:
#         print(beta)
#         start_time = time.perf_counter()
#         best_subgraph_obetas, best_surplus_avg_degree_obetas = g.greedy_obetas(beta)
#         end_time = time.perf_counter()
#         execution_time_obetas = end_time - start_time
#         print("GreedyUβS time: ", execution_time_obetas)
#         print("Best subgraph from GreedyObetaS:")
#         g.print_subgraph(best_subgraph_obetas)
#         print("With surplus average degree:", best_surplus_avg_degree_obetas)
#         vertices_obetas, num_edge_obetas, eval_obetas = g.evaluation_metric(best_subgraph_obetas)
#         print("Evaluation metric: ", eval_obetas)


# print("--------------------------------------------------------")
# start_time = time.perf_counter()
# best_subgraph_BF, best_density_BF = g.brute_force_search()
# end_time = time.perf_counter()
# execution_time_BF = end_time - start_time
# print("Brute Force time: ", execution_time_BF)
# print("Best subgraph from Brute Force:")
# g.print_subgraph(best_subgraph_BF)
# print("With surplus average degree:", best_density_BF)
# eval_BF = g.evaluation_metric(best_subgraph_BF)
# print("Evaluation metric: ", eval_BF)
#
# print(float(execution_time_UDS), float(execution_time_ObetaS), float(execution_time_ObetaS_2), float(execution_time_BF))
