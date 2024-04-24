# GNN_tests on QM9

## 1. Dijkstra:

- Adds the computed shortest weighted path between 2 nodes on the <current> graph representation to the feature space (before the first transformation).
- The shortest pathways are calculated using Dijkstra's algorithm (implemented in networkx)
  (https://networkx.org/documentation/stable/_modules/networkx/algorithms/shortest_paths/weighted.html#dijkstra_path)
- The parameters (input sizes, feature construction, acceptance criterion, ..etc) are the same as below.

## 2. Normal

- A normal computation of MSE on training [0:10000] and test [10000:12000] sets of the QM9 databse for [:0] (dipole moment).
  (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9)
- Stopping criterion: MSE_{test} < 0.005 or 1 million iterations, whichever happens first. 
