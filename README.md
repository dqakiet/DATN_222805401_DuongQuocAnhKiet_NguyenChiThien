# Finding the Highly Reliable Densest Subgraph from an Uncertain Weighted Graph


## Installation

main.py \
This script allows users to input parameters and datasets in order to execute the appropriate graph mining method. The script enables the application of algorithms to specific to users data and provides the results.

## File Format

The data files must follow a specified format in order for our graph extraction program to function effectively. This format ensures that graph data is accurately processed and evaluated. The data file should be in text format (.txt) and formatted as follows: \
•	File Format: The data should be stored in a text file, with each row representing an edge or link on the graph. \
•	Header Line: The first line of the file is designated as the header line, which explains the contents of each column. \
•	Node Columns: The first and second columns represent the nodes on each edge in the graph. These columns should provide identifiers (such as names or numbers) for each node. \
•	Feature Columns: The columns between the second and last columns represent different features connected with each edge. These properties will be used to apply weights to the graph's edges. \
•	Probability Column: The file's last column must show the probability of each edge's existence in the uncertain weighted graph. \
•	Column Separator: Each column should be separated by a space (" "), ensuring that different information components are clearly defined.

## Usage

You can test with my program

```bash
# Use for UWDS solution

python main.py ./dataset/579138.protein.links.detailed.v12.0.txt 1 uwds 3 
# Or for BWDS solution

python main.py ./dataset/579138.protein.links.detailed.v12.0.txt 1 bwds 3 0.5

```

