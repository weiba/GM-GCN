# GM-GCN
GM-GCN：Prediction of cancer driver genes based on gene-miRNA regulatory network with graph convolutional model 

## Prerequisites
-Python>=3.7.0

-pytorch>=1.9.0

-Cuda>=11.1

## Getting Started
### preprocessing
1.To get the normalized adjacency matrix, you need to prepare the following files:
1)"%s.txt" is the edges' file. e.g., PR.txt: <P_node_idx>\t<R_node_idx>.
2)"%s.txt" is the nodes' file. e.g., P.txt: <P_node_idx>\t<node_name>,We represent node_name with numbers

run gen_adj.py

### pretrain
1.pan-cancer

path = "./data/pan-cancer"

run pretrain.py

2.specific cancer

path = "./data/cancer name" e.g.brca

run pretrain.py

### train
1.pan-cancer

path = "./data/pan-cancer"

run train.py

2.specific cancer

path = "./data/cancer name" e.g.brca

run train.py

### result
return the AUC and AUPRC of ten-fold cross-validation.