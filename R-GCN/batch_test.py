import torch
from tqdm import tqdm
import csv
import numpy as np
from time import gmtime,strftime


def test(data, logits, Ks):

    time_now = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    res_file_name =  time_now +'RGCN.csv'
    res_file_path = './data/' + res_file_name
    test_users = data.qtime_users
    n_test_users = len(test_users)
    item_batch = range(data.n_items)
    rate_batch = torch.matmul(logits['user'][test_users].detach(),
                                  torch.transpose(logits['item'][item_batch].detach(), 0, 1))
    u_batch_size = 1000
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0
    for u_batch_id in tqdm(range(n_user_batchs)):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        batch_rank = np.argsort(-rate_batch[start:end], axis=1)[:, :Ks]
        batch_user = test_users[start:end]
        org_batch_user = [data.mapped_userId_to_org_userId[mapped_uid] for mapped_uid in batch_user]
        map_to_org = lambda i: data.mapped_itemId_to_org_itemId[i]
        vectorized_map_to_org = np.vectorize(map_to_org)
        org_batch_rank = vectorized_map_to_org(batch_rank)
        batch_res = np.hstack((np.expand_dims(org_batch_user, -1),org_batch_rank))
        count += len(batch_res)
        with open(res_file_path, "a",newline='') as fd:
            csvWriter = csv.writer(fd)
            csvWriter.writerows(batch_res)
    assert count == n_test_users
