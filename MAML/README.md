# MAML for KDD debiasing
This model is modified based on official MAML paper open source repository
note that underexpose_train/underexpose_test dir should be prepared and move in root dir

# Dependencies
numpy
tqdm
argparse
scipy
pytorch
heapq
multiprocessing

# GPU support
cuda related codes are commented for local verification. If you want to run the code on GPU, uncomment codes related to cuda and device.

# How to run:
in the project dir, run python3 kdd_train.py, it will read phrase 0-6 files, epoch 20000 times on MAML and 10 epoches on training task, 12 epoches on fine tuning. Parameters changed from original paper. The generated dlg_result.csv can be submitted.

# Expected running time:
more than 12 hours

# Results:
score submitted=0.04
ndcg based on best model=0.2237
