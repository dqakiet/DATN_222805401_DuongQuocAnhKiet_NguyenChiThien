# Finding the Highly Reliable Densest Subgraph from an Uncertain Weighted Graph

## Getting started
 In this project you can run this tool to find the highly reliable densest subgraph from an Uncertain Weighted Graph.

## main.py

This script allows users to input parameters and datasets in order to execute the appropriate graph mining method. The script enables the application of algorithms to specific to users data and provides the results.

## File Format

The data files must follow a specified format in order for our graph extraction program to function effectively. This format ensures that graph data is accurately processed and evaluated. The data file should be in text format (.txt) and formatted as follows: \
•	File Format: The data should be stored in a text file, with each row representing an edge or link on the graph. \
•	Header Line: The first line of the file is designated as the header line, which explains the contents of each column. \
•	Node Columns: The first and second columns represent the nodes on each edge in the graph. These columns should provide identifiers (such as names or numbers) for each node. \
•	Feature Columns: The columns between the second and last columns represent different features connected with each edge. These properties will be used to apply weights to the graph's edges. \
•	Probability Column: The file's last column must show the probability of each edge's existence in the uncertain weighted graph. \
•	Column Separator: Each column should be separated by a space (" "), ensuring that different information components are clearly defined.

## How can run tool to find subgraph
Syntax for run solution
python main.py [filename] [data_type] [algorithm] [column_index] [k] [b] [lambda (use in GreedyBWDS-O)]

with k is number of subgraph for finding top-k 
b is bound parameter for BWDS, BWDS-D, BWDS-O algorithm
lambda is parameter for cost overlap

Run for UWDS solution
```bash
python main.py ./dataset/579138.protein.links.detailed.v12.0.txt 1 uwds 3 
```
Run for BWDS solution
```bash
python main.py ./dataset/579138.protein.links.detailed.v12.0.txt 1 bwds 3 0.5
```
Run for BWDS-D solution
```bash
python main.py ./dataset/579138.protein.links.detailed.v12.0.txt 1 bwds_disjoint 3 0.5
```
Run for BWDS-O solution
```bash
python main.py ./dataset/579138.protein.links.detailed.v12.0.txt 1 bwds_overlap 2 3 0.5 0.6
```

## Results
When the main.py script is executed with the required parameters, our program not only displays the results in the Command Line Interface (CLI), but also saves them in a separate file in the “results” folder. This dual strategy to result distribution is essential for both right away evaluation and long-term storage of research results. \
The result file methodically includes different aspects of the mining subgraph, including: \
•	The vertices of the subgraph. \
•	The edges connecting these vertices, in with their weights and probabilities. \
•	Metrics including subgraph density, dependability indices, and other important statistical measures have been computed

