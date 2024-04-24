# GNN_tests on QM9

## 1. Dijkstra:

- Adds the computed shortest pathways between 2 nodes on the <current> graph representation to the feature space.
- The shortest pathways are calculated using Dijkstra's algorithm (implemented in networkx).
- The parameters (input sizes, feature construction, acceptance criterioon, ..etc) are the same as below.

## 2. Normal

- A normal computation of MSE on training [0:10000] and test [10000:12000] sets of the QM9 databse for [:0] (dipole moment).
  (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9)
- Stopping criterion: MSE_{test} < 0.005 or 1 million iterations, whichever happens first. 
