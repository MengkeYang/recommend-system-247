# Improved Item-CF
This is the implementation of improved Item-CF algorithms.

#### Reference
> Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001. Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th international conference on World Wide Web (WWW ’01). Association for Computing Machinery, New York, NY, USA, 285–295. [DOI]: https://doi.org/10.1145/371920.372071


## Introduction
In this work, we improve the Item-CF baseline by combining Item-CF with LightGBM and user entropies, respectively.

## Environment Requirement
The code has been tested running under Python 3.7. The required packages are as follows:
* pandas == 1.0.1
* numpy == 1.18.1
* tqdm == 4.45.0
* scikit-learn == 0.22.1
* lightgbm == 2.3.0
* matplotlib == 3.1.3

## Item-CF combined with LightGBM
The algorithm is implemented with Jupyter Notebook.

## Item-CF combined with user entropies
**Important: dataset should be prepared first and data directory should be assigned as arguments
, and make sure the provided dataset structure is exactly same as the downloaded one
 (the underexpose_test_click-X/ directory is nested in the underexpose_test/ directory
  and the underexpose_test_click-X.csv is nested in the underexpose_test_click-X/ directory).**

The instruction of commands has been clearly stated in the codes (see the parser function in parser.py).
- Firstly, change the current working directory to Item-CF/
```
cd Item-CF/
```
- Example command (**Make sure the '/' character is added behind the input directory**)
```
python Item-CF_user_entropies.py --train_path underexpose_train/ --test_path underexpose_test/
```
#### Predictions
Predictions will be generated in file "pred.csv" under the Item-CF/ directory.


================================
