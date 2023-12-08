import numpy as np
import time
import itertools
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

    def add_edge(self, u: str, v: str, p: float) -> None:
        """
        Adds an edge to the graph with a specified probability.
        :param u: The name of the first vertex.
        :param v: The name of the second vertex.
        :param p: The probability of the edge connecting vertex u and v.
        """
        if u not in self.adj_list:
            self.add_vertex(u)
        if v not in self.adj_list:
            self.add_vertex(v)
        self.adj_list[u].append((v, p))
        self.adj_list[v].append((u, p))

    def remove_vertex(self, vertex: str, temp_adj_list: dict[str, list[tuple[str, float]]]) -> None:
        """
        Removes a vertex and its associated edges from a temporary adjacency list.
        :param vertex: The vertex to be removed.
        :param temp_adj_list: A temporary adjacency list from which the vertex is removed.
        """
        for u, _ in temp_adj_list[vertex]:
            temp_adj_list[u] = [(v, p) for v, p in temp_adj_list[u] if v != vertex]
        del temp_adj_list[vertex]

    def get_expected_degree(self, vertex: str, temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the expected degree of a vertex based on edge probabilities.
        :param vertex: The vertex whose expected degree is to be calculated.
        :param temp_adj_list: The adjacency list used for calculation.
        :return: The expected degree of the given vertex.
        """
        return sum(prob for _, prob in temp_adj_list[vertex])

    def get_surplus_degree(self, vertex: str, beta: float, temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the surplus degree of a vertex.
        :param vertex: The vertex whose surplus degree is to be calculated.
        :param beta: The  beta value.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The surplus degree of the given vertex.
        """
        return sum((prob - beta) for _, prob in temp_adj_list[vertex])

    def calculate_expected_density(self, vertices: set[str], temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the expected density of a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The expected density of the subgraph.
        """
        total_prob = sum(prob for v in vertices for u, prob in temp_adj_list[v] if u in vertices and u > v)
        return total_prob / len(vertices) if vertices else 0

    def calculate_surplus_average_degree(self, vertices: set[str], beta: float, temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
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
            for u, prob in temp_adj_list[v]:
                if u in vertices and u > v:
                    total_prob += prob
                    num_edges += 1
        return (total_prob - beta*(num_edges)) / len(vertices) if num_edges > 0 else 0

    def greedyUDS(self) -> tuple[set[str], float]:
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

    def greedyOβS(self, beta: float) -> tuple[set[str], float]:
        """
        Applies a greedy algorithm to find the subgraph with the highest surplus average degree.
        :param beta: The beta value.
        :return:  A tuple containing the set of vertices forming the best subgraph and its surplus average degree.
        """
        temp_adj_list = {v: edges.copy() for v, edges in self.adj_list.items()}
        vertices = set(self.adj_list.keys())
        best_subgraph = set()
        best_surplus_avg_degree = -float('inf')

        while vertices:
            queue = [(self.get_surplus_degree(v, beta, temp_adj_list), v) for v in vertices]
            queue.sort()
            current_surplus_avg_degree = self.calculate_surplus_average_degree(vertices, beta, temp_adj_list)
            if current_surplus_avg_degree > best_surplus_avg_degree:
                best_surplus_avg_degree = current_surplus_avg_degree
                best_subgraph = set(vertices)
            _, v_to_remove = queue.pop(0)
            self.remove_vertex(v_to_remove, temp_adj_list)
            vertices.remove(v_to_remove)

        return best_subgraph, round(best_surplus_avg_degree, 3)

    def brute_force_search(self):
        """
        Performs a brute-force search to find the subgraph with the highest expected edge density.
        :return: A tuple containing the set of vertices forming the best subgraph and its expected density.
        """
        best_subgraph = set()
        best_expected_density = 0
        temp_adj_list = {v: edges.copy() for v, edges in self.adj_list.items()}
        vertices = list(self.adj_list.keys())
        all_combinations = []
        for r in range(1, len(vertices) + 1):
            all_combinations.extend(itertools.combinations(vertices, r))

        for subgraph in all_combinations:
            subgraph_set = set(subgraph)
            expected_density = self.calculate_expected_density(subgraph_set, temp_adj_list)
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
        reliability = 1
        for v in vertices:
            for u, prob in self.adj_list[v]:
                if u in vertices and u > v:
                    reliability *= prob
        return reliability

    def average_edge_probability(self, vertices: set[str]) -> float:
        """
        Calculates the average edge probability of a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :return: The average edge probability of the subgraph.
        """
        total_prob = 0
        num_edges = 0
        for v in vertices:
            for u, prob in self.adj_list[v]:
                if u in vertices and u > v:
                    total_prob += prob
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
        total_prob = sum(prob for v in vertices for u, prob in self.adj_list[v] if u in vertices and u > v)
        max_edges = len(vertices) * (len(vertices) - 1) / 2
        return total_prob / max_edges

    def edge_probability_std(self, vertices: set[str]) -> float:
        """
        Calculates the standard deviation of edge probabilities in a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :return: The standard deviation of edge probabilities in the subgraph.
        """
        edge_probs = [prob for v in vertices for u, prob in self.adj_list[v] if u in vertices and u > v]
        return np.std(edge_probs) if edge_probs else 0

    def evaluation_metric(self, subgraph: set[str]) -> dict[str, float]:
        """
        Evaluates a subgraph using various metrics.
        :param subgraph: A set of vertices forming the subgraph.
        :return: A dictionary of various evaluation metrics for the subgraph.
        """
        eed = self.expected_edge_density(subgraph)
        aep = self.average_edge_probability(subgraph)
        eps = self.edge_probability_std(subgraph)
        ar = self.adjoint_reliability(subgraph)

        return {
            'Expected edge density': round(eed, 3),
            'Average edge probability': round(aep, 3),
            'Edge probability std': round(eps, 3),
            'Adjoint Reliability': round(ar, 3),
        }

    def print_subgraph(self, vertices: set[str]) -> None:
        """
        Prints details of a subgraph including its vertices and edges.
        :param vertices: A set of vertices forming the subgraph.
        """
        print("Subgraph vertices (Total: {}):".format(len(vertices)), vertices)
        print("Edges:")
        printed_edges = set()
        for vertex in vertices:
            for neighbor, prob in self.adj_list[vertex]:
                if neighbor in vertices and (neighbor, vertex) not in printed_edges:
                    print(f"({vertex}, {neighbor}): {prob}")
                    printed_edges.add((vertex, neighbor))


# test
g = Graph()
g.add_edge('A', 'B', 0.8)
g.add_edge('A', 'C', 0.8)
g.add_edge('A', 'D', 0.5)
g.add_edge('B', 'D', 0.6)
g.add_edge('B', 'E', 0.2)
g.add_edge('B', 'C', 0.3)
g.add_edge('C', 'D', 0.4)
g.add_edge('C', 'E', 0.9)
g.add_edge('D', 'F', 0.3)
g.add_edge('E', 'F', 0.7)

start_time = time.perf_counter()
best_subgraph_UDS, best_density_UDS = g.greedyUDS()
end_time = time.perf_counter()
execution_time_UDS = end_time - start_time
print("GreedyUDS time: ", execution_time_UDS)
print("Best subgraph from GreedyUDS:")
g.print_subgraph(best_subgraph_UDS)
print("With expected density:", best_density_UDS)
eval_UDS = g.evaluation_metric(best_subgraph_UDS)
print("Evaluation metric: ", eval_UDS)

print("--------------------------------------------------------")
beta = 0.5
start_time = time.perf_counter()
best_subgraph_OβS, best_surplus_avg_degree_OβS = g.greedyOβS(beta)
end_time = time.perf_counter()
execution_time_OβS = end_time - start_time
print("GreedyUβS time: ", execution_time_OβS)
print("Best subgraph from GreedyOβS:")
g.print_subgraph(best_subgraph_OβS)
print("With surplus average degree:", best_surplus_avg_degree_OβS)
eval_UβS = g.evaluation_metric(best_subgraph_OβS)
print("Evaluation metric: ", eval_UβS)

print("--------------------------------------------------------")
start_time = time.perf_counter()
best_subgraph_BF, best_density_BF = g.brute_force_search()
end_time = time.perf_counter()
execution_time_BF = end_time - start_time
print("Brute Force time: ", execution_time_BF)
print("Best subgraph from Brute Force:")
g.print_subgraph(best_subgraph_BF)
print("With surplus average degree:", best_density_BF)
eval_BF = g.evaluation_metric(best_subgraph_BF)
print("Evaluation metric: ", eval_BF)

print(float(execution_time_UDS), float(execution_time_OβS), float(execution_time_BF))
