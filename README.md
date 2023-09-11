# CSE8803_Final_Project
### Pic2Graph: Graph-based Image Classification of Breast Carcinoma Subtypes

A project that: 
⋅⋅* Converted H&E stained histology images to cell graphs via residual network.9
⋅⋅* Applied residual network and multiple graph-based deep learning models, such as Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), GraphSAGE, Graph Isomorphism Network (GIN), to classify breast carcinoma subtypes in BReAst Carcinoma Subtyping (BRACS) dataset.

Detailed descriptions and results are in final report: MLB_final_report.pdf

BRACS Dataset: <https://www.bracs.icar.cnr.it/>

Cell graph generation: cell_graph.ipynb

File name explanation: 
⋅⋅⋅ File starts with the name of the neural network, such as CNN, GAT, GCN, GNN, GraphSAGE.
⋅⋅⋅ Binary vs. 7classes: classification tasks with binary classes (normal vs. tumor) or seven categories of tumors.
⋅⋅⋅ Edge weight: add edge features to cell graph.
