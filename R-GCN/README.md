# R-GCN model
* Paper: Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European Semantic Web Conference. Springer, Cham, 2018.[Paper in arXiv](https://arxiv.org/pdf/1703.06103.pdf)
* Code inspired by: [DGL_recsys](https://github.com/hoangdzung/DGL_recsys)

## Environment Requirement
The code has been tested running under Python 3.7. The required packages are as follows:
* argh == 0.26.2
* dgl == 0.4.3
* heapdict == 1.0.1
* numpy == 1.18.1
* pandas == 1.0.1
* pytorch == 1.5.0
* scipy == 1.4.1
* tqdm == 4.42.1

## Input files 
Besides the datasets from KDD, you will also need the following files for running our model:
* item_list.txt: Mapping which makes unique item ids sequential (0 to num_item)
* user_list.txt: Mapping which makes unique user ids sequential (0 to num_user)

## How to run
```
python3 main.py --train_data_dir <path/to/train_data/folder> --test_data_dir <path/to/test_data/folder> --map_data_dir <path/to/map_id/folder> --feature_data_dir <path/to/feature_data/folder> 
```

## KDD LB Scores
200 epoches, stopped before convergence 
* @hitrate50full: 0.317
* @ndcg50full: 0.088  
* @hitrate50half: 0.220 
* @ndcg50half: 0.060

