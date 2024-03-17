from weighted import Graph
import time

def read_file(column: str) -> tuple[str, str, int, int]:
    """

    :param column: Column of feature to create graph
    :return: A Tuple with 2 vertices, weight and probability of weight
    """
    u = []
    v = []
    w = []
    p = []
    with open('./dataset/579138.protein.links.detailed.v12.0.txt', 'r') as f:
        lines = f.readlines()
        header = lines[0].split()
        for line in lines[1:]:
            parts = line.split()
            u.append(parts[0])
            v.append(parts[1])
            w.append(float(parts[column]))
            p.append(parts[-1])
    return u, v, w, p


def initial_graph_feature_protein(feature_column: int) -> dict[str, list[tuple[str, float]]]:
    """

    :param feature_column: A dictionary to test with column and feature name in dataset
    :return: A Graph fit for test protein dataset with initial weight is 0.041 * 1000 for value 0 in dataset and probaility for weight must divide 1000
    """

    g = Graph()

    print(index[feature_column])
    u, v, w, p = read_file(feature_column)
    for j in range(len(u)):
        if w[j] == 0.0:
            w[j] = 0.041 * 1000
        g.add_edge(u[j], v[j], w[j], float(p[j]) / 1000)
        # g.print_summarize_graph()     # Print summary graph of feature
    return g


def get_uwds_algorithm(g: Graph) -> None:
    """

    :param g: A Graph to mining
    :return:  Print subgraph vertices, number of edges , expected density, execution time and evaluation metric by uwds algorithm
    """
    start_time = time.perf_counter()
    best_subgraph_uwds, best_density_uwds = g.greedy_uwds()
    end_time = time.perf_counter()
    execution_time_uwds = end_time - start_time
    print("GreedyUWDS time: ", execution_time_uwds)
    print("Best subgraph from GreedyUWDS:")
    g.print_subgraph(best_subgraph_uwds)
    print("With weighted expected density:", best_density_uwds)
    vertices_uwds, num_edge_uwds, eval_uwds = g.evaluation_metric(best_subgraph_uwds)
    print("Evaluation metric: ", eval_uwds)


def get_bwds_algorithm(g : Graph, bounds: list[int]) -> None:
    """

    :param g: A Graph to mining
    :param bounds: Parameter bounds for algorithm
    :return: Print subgraph vertices, number of edges , expected density, execution time and evaluation metric by uwds algorithm
    """
    for b in bounds:
        print(b)
        start_time = time.perf_counter()
        best_subgraph_bwds, best_excess_avg_degree_bwds = g.greedy_bwds(b)
        end_time = time.perf_counter()
        execution_time_bwds = end_time - start_time
        print("GreedyBWDS time: ", execution_time_bwds)
        print("Best subgraph from Greedybwds:")
        g.print_subgraph(best_subgraph_bwds)
        print("With excess average degree:", best_excess_avg_degree_bwds)
        vertices_bwds, num_edge_bwds, eval_bwds = g.evaluation_metric(best_subgraph_bwds)
        print("Evaluation metric: ", eval_bwds)


if __name__ == '__main__':
    # Initial bound to test in dataset
    bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

    # A dictionary with key is number of column in dataset and value is name of feature in protein dataset
    index = {2: "neighborhood", 3: "fusion", 4: "cooccurence", 5: "coexpression", 6: "experimental", 7: "database",
             8: "textmining"}

    for i in index.keys():
        g = initial_graph_feature_protein(i)

        # UWDS algorithm
        get_uwds_algorithm(g)

        # BWDS algorithm
        get_bwds_algorithm(g, bounds)
