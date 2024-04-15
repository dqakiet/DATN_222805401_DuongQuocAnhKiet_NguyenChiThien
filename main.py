from weighted import Graph
import time
import sys
import os

def read_file(filename: str, column_index: int) -> tuple[list[str], list[str], list[str], list[float], list[str]]:
    """
    :param filename: Name of the file to read
    :param column: Column of feature to create graph
    :return: Header and a Tuple with 2 vertices, weight and probability of weight
    """
    u = []
    v = []
    w = []
    p = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        header = lines[0].split()
        for line in lines[1:]:
            parts = line.split()
            u.append(parts[0])
            v.append(parts[1])
            w.append(float(parts[column_index]))
            p.append(parts[-1])
    return header, u, v, w, p

def get_header(filename:str):
    with open(filename,'r') as f:
        lines = f.readlines()
        header = lines[0].split()
        return header, header[2:-1]

def initial_graph_feature_non_protein(filename: str, column_index: int) -> dict[str, list[tuple[str, float]]]:
    """

    :param filename: File to read dataset
    :param column_index: Index of feature to analyze
    :return: A Graph for test non-protein dataset
    """
    g = Graph()
    header, u, v, w, p = read_file(filename, column_index)

    for j in range(len(u)):
        g.add_edge(u[j], v[j], w[j], float(p[j]))
    return g, header


def initial_graph_feature_protein(filename: str, column_index: int) -> dict[str, list[tuple[str, float]]]:
    """
    :param feature_column: Column index for the feature in dataset
    :param colume_index: Index of feature to analyze
    :return: A Graph fit for test protein dataset with initial weight is 0.041 * 1000 for value 0 in dataset and probability for weight must divide 1000
    """

    g = Graph()
    header, u, v, w, p = read_file(filename, column_index)
    for j in range(len(u)):
        if w[j] == 0.0:
            w[j] = 0.041 * 1000
        g.add_edge(u[j], v[j], w[j], float(p[j]) / 1000)
    return g, header


def get_uwds_algorithm(g: Graph, header: list, column_index: int) -> None:
    """

    :param g: A Graph to mining
    :param header: A ist of name feature
    :param column_index: Index of feature to analyze
    :return: Print all information about the subgraph found by uwds algorithm
    """
    start_time = time.perf_counter()
    best_subgraph_uwds, best_density_uwds = g.greedy_uwds()
    end_time = time.perf_counter()
    execution_time_uwds = end_time - start_time
    vertices_uwds, subgraph_dict_uwds, num_edge_uwds, eval_uwds = g.evaluation_metric(best_subgraph_uwds)
    output_str = "Feature: " + str(header[column_index]) + "\n"
    output_str += "======================================================\n"
    output_str += "GreedyUWDS time: " + str(execution_time_uwds) + "\n"
    output_str += "Best subgraph from GreedyUWDS:" + "\n"
    output_str += "With weighted expected density: " + str(best_density_uwds) + "\n"
    output_str += "Subgraph vertices (Total: {}):".format(len(subgraph_dict_uwds)) + "\n"
    output_str += str(subgraph_dict_uwds) + "\n\n"
    output_str += "======================================================\n"
    output_str += "Number of edges: " + str(num_edge_uwds) + "\n"
    for key, value in eval_uwds.items():
        output_str += str(key) + ": " + str(value) + "\n"
    output_str += "======================================================\n"
    print(output_str)
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)
    write_to_file(output_str, f"./results/uwds_results_{column_index}_{header[column_index]}.txt")

def get_bruteforce_algorithm(g: Graph, header: list, column_index: int) -> None:
    """

    :param g: A Graph to mining
    :param header: A ist of name feature
    :param column_index: Index of feature to analyze
    :return: Print all information about the subgraph found by brute force algorithm
    """
    start_time = time.perf_counter()
    best_subgraph_bf, best_density_bf = g.brute_force_search()
    end_time = time.perf_counter()
    execution_time_bf = end_time - start_time
    vertices_bf, subgraph_dict_bf, num_edge_bf, eval_bf = g.evaluation_metric(best_subgraph_bf)
    output_str = "Feature: " + str(header[column_index]) + "\n"
    output_str += "======================================================\n"
    output_str += "Brute Force time: " + str(execution_time_bf) + "\n"
    output_str += "Best subgraph from Brute Force: " + "\n"
    output_str += "With weighted expected density: " + str(best_density_bf) + "\n"
    output_str += "Subgraph vertices (Total: {}):".format(len(subgraph_dict_bf)) + "\n"
    output_str += str(subgraph_dict_bf) + "\n\n"
    output_str += "======================================================\n"
    output_str += "Number of edges: " + str(num_edge_bf) + "\n"
    for key, value in eval_bf.items():
        output_str += str(key) + ": " + str(value) + "\n"
    output_str += "======================================================\n"
    print(output_str)
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)
    write_to_file(output_str, f"./results/bf_results_{column_index}_{header[column_index]}.txt")

def get_bwds_algorithm(g: Graph, b: int, header: list, column_index:int) -> None:
    """

    :param g: A Graph to mining
    :param bounds: Parameter bounds for algorithm
    :param header: A ist of name feature
    :param column_index: Index of feature to analyze
    :return: Print all information about the subgraph found by uwds algorithm
    """
    start_time = time.perf_counter()
    best_subgraph_bwds, best_excess_avg_degree_bwds = g.greedy_bwds(b)
    end_time = time.perf_counter()
    execution_time_bwds = end_time - start_time
    vertices_bwds, subgraph_dict_bwds, num_edge_bwds, eval_bwds = g.evaluation_metric(best_subgraph_bwds)
    output_str = "Feature: " + str(header[column_index]) + "\n"
    output_str += "======================================================\n"
    output_str += "GreedyBWDS time: " + str(execution_time_bwds) + "\n"
    output_str += "Bound: " + str(b) + "\n"
    output_str += "Best subgraph from GreedyBWDS:" + "\n"
    output_str += "With excess average degree: " + str(best_excess_avg_degree_bwds) + "\n"
    output_str += "Subgraph vertices (Total: {}):".format(len(subgraph_dict_bwds)) + "\n"
    output_str += str(subgraph_dict_bwds) + "\n\n"
    output_str += "======================================================\n"
    output_str += "Number of edges: " + str(num_edge_bwds) + "\n"
    for key, value in eval_bwds.items():
        output_str += str(key) + ": " + str(value) + "\n"
    output_str += "======================================================\n"
    print(output_str)
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)
    write_to_file(output_str, f"./results/bwds_results_{column_index}_{header[column_index]}_{b}.txt")


def write_to_file(data: str, filename: str) -> None:
    """

    :param data: String to write to file
    :param filename: File to save
    """
    with open(filename, 'w') as file:
        file.write(data + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(
            "Usage: [filename] [data type: 1 (protein) or 0 (non-protein)] [algorithm: UWDS or BWDS] [column index] [bounds for BWDS (optional)]")
        sys.exit(1)

    filename = sys.argv[1]
    data_type = int(sys.argv[2])
    algorithm = sys.argv[3]
    column_input = sys.argv[4]
    if '-' in column_input:
        start, end = map(int, column_input.split('-'))
        column_indices = range(start, end)
        header, header_tmp = get_header(filename)
        if len(column_indices) > len(header_tmp):
            print("Column index must be greater than or equal to 2 and less than", len(header) - 1)
            sys.exit(1)
    else:
        column_indices = [int(column_input)]
    for column_index in column_indices:
        if data_type == 1:
            g, header = initial_graph_feature_protein(filename, column_index)
        elif data_type == 0:
            g, header = initial_graph_feature_non_protein(filename, column_index)
        if algorithm == "UWDS" or algorithm == "uwds":
            get_uwds_algorithm(g, header, column_index)
        if algorithm == "BF" or algorithm == "bf":
            get_bruteforce_algorithm(g, header, column_index)
        elif algorithm == "BWDS" or algorithm == "bwds":
            if len(sys.argv) < 6:
                print("Please provide a bound value for BWDS algorithm.")
                sys.exit(1)
            try:
                bound = float(sys.argv[5])
            except ValueError:
                print("Bound must be a number.")
                sys.exit(1)
            get_bwds_algorithm(g, bound, header, column_index)

