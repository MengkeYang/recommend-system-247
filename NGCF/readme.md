# DGL_recsys modified for KDD recsys
note that underexpose_train/underexpose_test dir should be prepared in the root dir

# Dependencies
numpy
tqdm
pytorch
heapq
argparse
scipy
multiprocessing
cuda

# How to run:
in the project dir, run python3 main.py, it will read phrase 0-6 files, epoch 120 times using original parameters and 
generate dlg_result.csv that can be submitted.

# Expected running time:
12 hours

# Results:
score=0.14
