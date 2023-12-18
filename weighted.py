import numpy as np
import time
import itertools
import heapq

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

    def get_mean_weight(self) -> float:
        """
        Get average weight of initial graph
        :return: Average weight of initial graph
        """
        vertices = set(self.adj_list.keys())
        total_weight = sum(weight for v in vertices for u, weight, _ in self.adj_list[v] if u in vertices and u > v)
        num_edges = sum(1 for vertex in self.adj_list for _ in self.adj_list[vertex]) // 2
        return total_weight / num_edges if num_edges else 0

    def get_expected_degree(self, vertex: str, temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the expected degree of a vertex based on edge probabilities.
        :param vertex: The vertex whose expected degree is to be calculated.
        :param temp_adj_list: The adjacency list used for calculation.
        :return: The expected degree of the given vertex.
        """
        return sum((weight*prob) for _, weight, prob in temp_adj_list[vertex])

    def calculate_expected_density(self, vertices: set[str], temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the expected density of a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The expected density of the subgraph.
        """
        total_prob = sum((weight*prob) for v in vertices for u, weight, prob in temp_adj_list[v] if u in vertices and u > v)
        return total_prob / len(vertices) if vertices else 0

    ### \sum w(e)*p(e) - \beta *mean(w(e))
    def get_surplus_degree(self, vertex: str, beta: float, temp_adj_list: dict[str, list[tuple[str, float]]], avgweight: float) -> float:
        """
        Calculates the surplus degree of a vertex.
        :param vertex: The vertex whose surplus degree is to be calculated.
        :param beta: The  beta value.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The surplus degree of the given vertex.
        """
        return sum((weight*prob) - (beta*avgweight) for _, weight, prob in temp_adj_list[vertex])

    def calculate_surplus_average_degree(self, vertices: set[str], beta: float, temp_adj_list: dict[str, list[tuple[str, float]]], avgweight: float) -> float:
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
                    total_prob += prob*weight - beta * avgweight
                    num_edges += 1
        return (total_prob) / len(vertices) if num_edges > 0 else 0

    def get_surplus_degree_2(self, vertex: str, beta: float, temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
        """
        Calculates the surplus degree of a vertex.
        :param vertex: The vertex whose surplus degree is to be calculated.
        :param beta: The  beta value.
        :param temp_adj_list: The adjacency list used for the calculation.
        :return: The surplus degree of the given vertex.
        """
        return sum((weight*(prob - beta)) for _, weight, prob in temp_adj_list[vertex])

    def calculate_surplus_average_degree_2(self, vertices: set[str], beta: float, temp_adj_list: dict[str, list[tuple[str, float]]]) -> float:
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
                    total_prob += weight*(prob-beta)
                    num_edges += 1
        return (total_prob) / len(vertices) if num_edges > 0 else 0

    def greedyOβS(self, beta: float) -> tuple[set[str], float]:
        """
        Applies a greedy algorithm to find the subgraph with the highest surplus average degree.
        :param beta: The beta value.
        :return:  A tuple containing the set of vertices forming the best subgraph and its surplus average degree.
        """
        temp_adj_list = {v: edges.copy() for v, edges in self.adj_list.items()}
        vertices = set(self.adj_list.keys())
        best_subgraph = set()
        best_surplus_avg_degree = 0
        avgweight = self.get_mean_weight()

        heap = []
        vertex_to_marker = {}
        current_marker = 0

        for v in vertices:
            surplus_degree = self.get_surplus_degree(v, beta, temp_adj_list, avgweight)
            heapq.heappush(heap, (surplus_degree, v, current_marker))
            vertex_to_marker[v] = current_marker

        while len(vertices) >= 2:
            current_density = self.calculate_surplus_average_degree(vertices, beta, temp_adj_list, avgweight)
            if current_density > best_surplus_avg_degree:
                best_surplus_avg_degree = current_density
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
                    new_degree = self.get_surplus_degree(neighbor, beta, temp_adj_list, avgweight)
                    heapq.heappush(heap, (new_degree, neighbor, current_marker))
                    vertex_to_marker[neighbor] = current_marker

        return best_subgraph, round(best_surplus_avg_degree, 3)

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

    def greedyOβS_2(self, beta: float) -> tuple[set[str], float]:
        """
        Applies a greedy algorithm to find the subgraph with the highest surplus average degree.
        :param beta: The beta value.
        :return:  A tuple containing the set of vertices forming the best subgraph and its surplus average degree.
        """
        temp_adj_list = {v: edges.copy() for v, edges in self.adj_list.items()}
        vertices = set(self.adj_list.keys())
        best_subgraph = set()
        best_surplus_avg_degree = 0
        avgweight = self.get_mean_weight()

        heap = []
        vertex_to_marker = {}
        current_marker = 0

        for v in vertices:
            surplus_degree = self.get_surplus_degree_2(v, beta, temp_adj_list)
            heapq.heappush(heap, (surplus_degree, v, current_marker))
            vertex_to_marker[v] = current_marker

        while len(vertices) >= 2:
            current_density = self.calculate_surplus_average_degree_2(vertices, beta, temp_adj_list)
            if current_density > best_surplus_avg_degree:
                best_surplus_avg_degree = current_density
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
                    new_degree = self.get_surplus_degree(neighbor, beta, temp_adj_list, avgweight)
                    heapq.heappush(heap, (new_degree, neighbor, current_marker))
                    vertex_to_marker[neighbor] = current_marker

        return best_subgraph, round(best_surplus_avg_degree, 3)

    def brute_force_search(self) -> tuple[set[str], float]:
        best_subgraph = set()
        best_expected_density = 0
        vertices = list(self.adj_list.keys())

        # Xác định tất cả các cạnh
        edges = set()
        for vertex in vertices:
            for neighbor, _, _ in self.adj_list[vertex]:
                if (vertex, neighbor) not in edges and (neighbor, vertex) not in edges:
                    edges.add((vertex, neighbor))
        # Xem xét tất cả các tập hợp con của các cạnh
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
                    reliability += np.log10(prob)
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
            for u, weight, prob in self.adj_list[v]:
                if u in vertices and u > v:
                    total_prob += prob*weight
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
        total_prob = sum((weight*prob) for v in vertices for u, weight, prob in self.adj_list[v] if u in vertices and u > v)
        max_edges = len(vertices) * (len(vertices) - 1) / 2
        return total_prob / max_edges

    def edge_probability_std(self, vertices: set[str]) -> float:
        """
        Calculates the standard deviation of edge probabilities in a subgraph.
        :param vertices: A set of vertices forming the subgraph.
        :return: The standard deviation of edge probabilities in the subgraph.
        """
        edge_probs = [(weight*prob) for v in vertices for u, weight, prob in self.adj_list[v] if u in vertices and u > v]
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
        num_edges = 0
        print("Subgraph vertices (Total: {}):".format(len(vertices)), vertices)
        for v in vertices:
            for u, _, _ in self.adj_list[v]:
                if u in vertices and u > v:
                    num_edges += 1
        print("Num Edges:", num_edges)
        # printed_edges = set()
        # for vertex in vertices:
        #     for neighbor, weight, prob in self.adj_list[vertex]:
        #         if neighbor in vertices and (neighbor, vertex) not in printed_edges:
        #             print(f"({vertex}, {neighbor}): {weight} {prob}")
        #             printed_edges.add((vertex, neighbor))


# test
g = Graph()
# g.add_edge('A', 'B', 40, 0.8)
# g.add_edge('A', 'C', 10, 0.1)
# g.add_edge('A', 'D', 70, 0.7)
# g.add_edge('A', 'E', 90, 0.9)
# g.add_edge('B', 'D', 20, 0.8)
# g.add_edge('B', 'E', 50, 0.4)
# g.add_edge('B', 'F', 10, 0.9)
# g.add_edge('C', 'D', 20, 0.3)
# g.add_edge('C', 'F', 30, 0.3)
# g.add_edge('D', 'E', 80, 0.2)
# g.add_edge('E', 'F', 60, 0.4)

def read_file():
    u = []
    v = []
    w = []
    p = []
    with open('579138.protein.links.detailed.v12.0.txt', 'r') as f:
        lines = f.readlines()
        header = lines[0].split()
        for line in lines[1:]:
            parts = line.split()
            u.append(parts[0])
            v.append(parts[1])
            w.append(float(parts[2]))
            p.append(parts[-1])
    return u, v, w, p


u, v, w, p = read_file()
for i in range(len(u)):
    if w[i] == 0.0:
        w[i] = 0.041*1000

    g.add_edge(u[i],v[i],w[i], int(p[i])/1000)


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
beta = 0.3
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
best_subgraph_OβS, best_surplus_avg_degree_OβS = g.greedyOβS_2(beta)
end_time = time.perf_counter()
execution_time_OβS_2 = end_time - start_time
print("GreedyUβS_2 time: ", execution_time_OβS_2)
print("Best subgraph from GreedyOβS_2:")
g.print_subgraph(best_subgraph_OβS)
print("With surplus average degree:", best_surplus_avg_degree_OβS)
eval_UβS = g.evaluation_metric(best_subgraph_OβS)
print("Evaluation metric_2: ", eval_UβS)
print(float(execution_time_UDS), float(execution_time_OβS), float(execution_time_OβS_2))

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
# print(float(execution_time_UDS), float(execution_time_OβS), float(execution_time_OβS_2), float(execution_time_BF))
