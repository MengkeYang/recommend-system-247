# LightGCN
This is the implementation of Light-GCN

#### Reference
> Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).


## Introduction
In this work, NGCF algorithm is improved by eliminating redundant components.

## Environment Requirement
The code has been tested running under Python 3.7. The required packages are as follows:
* tensorflow == 1.11.0
* numpy == 1.18.1
* scipy == 1.4.1
* sklearn == 0.19.1
* Cython == 0.29.19
* pandas == 1.0.1

## C++ evaluator
C++ evaluator is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
After compilation, the C++ evaluator will run by default instead of Python evaluator. If the compilation fails due to platform issues, Python evaluator will be adopted.

## How to run
**Important: dataset should be prepared first and data directory should be assigned as arguments
, and make sure the provided dataset structure is exactly same as the downloaded one
 (the underexpose_test_click-X/ directory is nested in the underexpose_test/ directory
  and the underexpose_test_click-X.csv is nested in the underexpose_test_click-X/ directory).**

- Firstly, change the current working directory to light-gcn/
```
cd light-gcn/
```

- Secondly, run preprocess.py to preprocess the raw dataset

**The instruction of commands has been clearly stated in the codes (see the parse_args() function in preprocess.py).**

```
python preprocess.py --train_path underexpose_train/ --test_path underexpose_test/
```

The instruction of commands has been clearly stated in the codes (see the parser function in utility/parser.py).

- Example command
```
python LightGCN.py --dataset kdd --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 4096 --epoch 20
```

* Output log:
```
eval_score_matrix_foldout with cpp
n_users=31525, n_items=98769
n_interactions=966669
n_train=760863, n_test=205806, sparsity=0.00031
already create adjacency matrix (130294, 130294) 323.98045802116394
generate single-normalized adjacency matrix.
    ...
```
NOTE : the duration of training and testing depends on the running environment.

## Dataset after preprocessing
* `train.txt`
  * Train file.
  * Each line is a user with his/her positive interactions with items: userID\t a list of itemID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with his/her positive interactions with items: userID\t a list of itemID\n.
  * Here, all unobserved interactions are treated as the negative instances when reporting performance.
  
* `user_list.txt`
  * User IDs.
  * Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and preprocessed dataset, respectively.
  
* `item_list.txt`
  * Item IDs.
  * Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and preprocessed dataset, respectively.
  
## Predictions
Predictions will be generated in file "pred.csv" under the light-gcn/ directory.

================================
