import numpy as np
import time

class Graph:
    def __init__(self) -> None:
        self.adj_list: dict[str, list[tuple[str, float]]] = {}

    def add_vertex(self, vertex: str) -> None:
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, u: str, v: str, p: float) -> None:
        if u not in self.adj_list:
            self.add_vertex(u)
        if v not in self.adj_list:
            self.add_vertex(v)
        self.adj_list[u].append((v, p))
        self.adj_list[v].append((u, p))

    def get_expected_degree(self, vertex: str) -> float:
        return sum(prob for _, prob in self.adj_list[vertex])

    def get_surplus_degree(self, vertex: str, beta: float) -> float:
        return sum(prob - beta for _, prob in self.adj_list[vertex])

    def calculate_expected_density(self, vertices: set[str]) -> float:
        total_prob = sum(prob for v in vertices for u, prob in self.adj_list[v] if u in vertices and u > v)
        return total_prob / len(vertices) if vertices else 0

    def calculate_surplus_average_degree(self, vertices: set[str], beta: float) -> float:
        total_surplus = 0
        counted_edges = set()
        for v in vertices:
            for u, prob in self.adj_list[v]:
                if u in vertices:
                    edge = tuple(sorted((u, v)))
                    if edge not in counted_edges:
                        total_surplus += (prob - beta)
                        counted_edges.add(edge)
        return total_surplus / len(vertices) if vertices else 0

    def greedyUDS(self) -> tuple[set[str], float]:
        vertices = list(self.adj_list.keys())
        best_subgraph = set()
        best_density = 0

        for i in range(len(vertices), 1, -1):
            current_density = self.calculate_expected_density(set(vertices))
            if current_density > best_density:
                best_density = current_density
                best_subgraph = set(vertices)
            v = min(vertices, key=lambda x: self.get_expected_degree(x))
            vertices.remove(v)

        return best_subgraph, round(best_density,3)

    def greedyOβS(self, beta: float) -> tuple[set[str], float]:
        vertices = set(self.adj_list.keys())
        best_subgraph = set()
        best_surplus_avg_degree = 0

        queue = [(self.get_surplus_degree(v, beta), v) for v in vertices]
        queue.sort()

        while len(queue) > 1:
            current_surplus_avg_degree = self.calculate_surplus_average_degree(vertices, beta)
            if current_surplus_avg_degree > best_surplus_avg_degree:
                best_surplus_avg_degree = current_surplus_avg_degree
                best_subgraph = set(vertices)
            _, v = queue.pop(0)
            vertices.remove(v)
            queue = [(self.get_surplus_degree(v, beta), v) for v in vertices]
            queue.sort()

        return best_subgraph, round(best_surplus_avg_degree,3)

    def adjoint_reliability(self, vertices: set[str]) -> float:
        reliability = 1
        for v in vertices:
            for u, prob in self.adj_list[v]:
                if u in vertices and u > v:
                    reliability *= prob
        return reliability

    def average_edge_probability(self, vertices: set[str]) -> float:
        total_prob = 0
        num_edges = 0
        for v in vertices:
            for u, prob in self.adj_list[v]:
                if u in vertices and u > v:
                    total_prob += prob
                    num_edges += 1
        return total_prob / num_edges if num_edges > 0 else 0

    def expected_edge_density(self, vertices: set[str]) -> float:
        total_prob = sum(prob for v in vertices for u, prob in self.adj_list[v] if u in vertices and u > v)
        max_edges = len(vertices) * (len(vertices) - 1) / 2
        return total_prob / max_edges if vertices else 0

    def edge_probability_std(self, vertices: set[str]) -> float:
        edge_probs = [prob for v in vertices for u, prob in self.adj_list[v] if u in vertices and u > v]
        return np.std(edge_probs) if edge_probs else 0

    def evaluation_metric(self, subgraph: set[str]) -> dict[str, float]:
        eed = self.expected_edge_density(subgraph)
        aep = self.average_edge_probability(subgraph)
        eps = self.edge_probability_std(subgraph)
        ar = self.adjoint_reliability(subgraph)

        return {
            'Expected edge density': round(eed,3),
            'Average edge probability': round(aep,3),
            'Edge probability std': round(eps,3),
            'Adjoint Reliability': round(ar,3),
        }

    def print_subgraph(self, vertices: set[str]) -> None:
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
g.add_edge('A', 'B', 0.9)
g.add_edge('A', 'C', 0.7)
g.add_edge('B', 'C', 0.8)
g.add_edge('B', 'D', 0.5)
g.add_edge('C', 'E', 0.6)
g.add_edge('D', 'E', 0.7)
g.add_edge('D', 'F', 0.3)
g.add_edge('E', 'F', 0.1)
g.add_edge('F', 'G', 0.2)


best_subgraph_UDS, best_density_UDS = g.greedyUDS()
print("Best subgraph from GreedyUDS:")
g.print_subgraph(best_subgraph_UDS)
print("With expected density:", best_density_UDS)
eval_UDS = g.evaluation_metric(best_subgraph_UDS)
print("Evaluation metric: ", eval_UDS)

beta = 0.7
best_subgraph_OβS, best_surplus_avg_degree_OβS = g.greedyOβS(beta)
print("\nBest subgraph from GreedyOβS:")
g.print_subgraph(best_subgraph_OβS)
print("With surplus average degree:", best_surplus_avg_degree_OβS)
eval_UβS = g.evaluation_metric(best_subgraph_OβS)
print("Evaluation metric: ", eval_UβS)

# def read_file():
#     u = []
#     v = []
#     p = []
#     with open('579138.protein.links.detailed.v12.0.txt', 'r') as f:
#         lines = f.readlines()
#         header = lines[0].split()
#         for line in lines[1:]:
#             parts = line.split()
#             u.append(parts[0])
#             v.append(parts[1])
#             p.append(parts[-1])
#     return u, v, p
#
#
# u, v, p = read_file()
# for i in range(len(u)):
#     g.add_edge(u[i],v[i],int(p[i])/1000)
#
# start_time = time.time()
# best_subgraph_UDS, best_density_UDS = g.greedyUDS()
# end_time = time.time()
# execution_time = end_time - start_time
# print(execution_time)
# print("Best subgraph from GreedyUDS:")
# g.print_subgraph(best_subgraph_UDS)
# print("With expected density:", best_density_UDS)
#
# beta = 0.5
# start_time = time.time()
# best_subgraph_OβS, best_surplus_avg_degree_OβS = g.greedyOβS(beta)
# end_time = time.time()
# execution_time = end_time - start_time
# print(execution_time)
# print("\nBest subgraph from GreedyOβS:")
# g.print_subgraph(best_subgraph_OβS)
# print("With surplus average degree:", best_surplus_avg_degree_OβS)
