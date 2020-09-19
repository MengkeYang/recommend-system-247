import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import os
import csv

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.n_users, self.n_items = 0, 0
        self.n_train = 0
        self.exist_users = []
        self.cur_phrase = 1
        self.old_new_users = {}
        self.old_new_items = {}
        self.new_old_items = {}
        self.index = 0

        # collect all training file names
        file_list = []
        for phrase in range(self.cur_phrase):
            file_name = os.path.join(path, 'underexpose_train_click-'+str(phrase)+'.csv')
            assert os.path.isfile(file_name)
            file_list.append(file_name)

            file_name = os.path.join(path, 'underexpose_test_click-'+str(phrase)+'.csv')
            assert os.path.isfile(file_name)
            file_list.append(file_name)

        for train_file in file_list:
            for line in open(train_file):
                line = list(line.strip().split(','))
                old_user_id = int(line[0])
                old_item_id = int(line[1])

                # convert old user id into continuous user id
                if old_user_id not in self.old_new_users:
                    self.old_new_users[old_user_id] = len(self.old_new_users)
                    self.exist_users.append(self.old_new_users[old_user_id])
                # convert old item id into continuous item id
                if old_item_id not in self.old_new_items:
                    self.old_new_items[old_item_id] = len(self.old_new_items)
                    self.new_old_items[self.old_new_items[old_item_id]] = old_item_id
                self.n_train += 1

        self.n_items = len(self.old_new_items)
        self.n_users = len(self.old_new_users)
        self.print_statistics()
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items = {}
        for train_file in file_list:
            for line in open(train_file):
                line = list(line.strip().split(','))
                old_user_id = int(line[0])
                old_item_id = int(line[1])
                user_id = self.old_new_users[old_user_id]
                item_id = self.old_new_items[old_item_id]
                self.R[user_id, item_id] = 1.
                if user_id in self.train_items:
                    self.train_items[user_id].append(item_id)
                else:
                    self.train_items[user_id] = [item_id]
        
        self.exist_test_users = []
        for phrase in range(self.cur_phrase):
            file_name = os.path.join("../underexpose_test/", 'underexpose_test_qtime-'+str(phrase)+'.csv')
            assert os.path.isfile(file_name)
            for line in open(file_name):
                line = list(line.strip().split(','))
                old_user_id = int(line[0])
                self.exist_test_users.append(self.old_new_users[old_user_id])

    # based on result, rearrange format and 
    # generate all answer until phrase T
    def generate_answer(self, result):
        # {user0: [top_50 items], user1: [top_50 items]}
        arranged_result = {}
        for batch_result in result:
            for top_50_map in batch_result:
                key = list(top_50_map.keys())[0]
                arranged_result[key] = top_50_map[key]

        # convert new index to old discrete index
        # and copy from arranged_result
        old_arranged_result = {}
        for phrase in range(self.cur_phrase):
            file_name = os.path.join("../underexpose_test/", 'underexpose_test_qtime-'+str(phrase)+'.csv')
            assert os.path.isfile(file_name)
            for line in open(file_name):
                line = list(line.strip().split(','))
                old_user_id = int(line[0])
                new_user_id = self.old_new_users[old_user_id]
                old_arranged_result[old_user_id] = []
                for new_item_reco in arranged_result[new_user_id]:
                    old_arranged_result[old_user_id].append(self.new_old_items[new_item_reco])

        # write to dgl_result.csv
        with open('dgl_result.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for old_user_id in old_arranged_result.keys():
                writer.writerow([old_user_id] + old_arranged_result[old_user_id])


    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train))
        print('n_train=%d, sparsity=%.5f' % (self.n_train, (self.n_train)/(self.n_users * self.n_items)))


    def get_adj_mat(self):
        try:
            t1 = time()
            norm_adj_mat = sp.load_npz(self.path + 's_norm_adj_mat.npz')
            print('already load adj matrix', norm_adj_mat.shape, time() - t1)
        except Exception:
            norm_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + 's_norm_adj_mat.npz', norm_adj_mat)
        return norm_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            # D^-1 * A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

        print('already normalize adjacency matrix', time() - t2)
        return norm_adj_mat.tocsr()


    # choose num items that u clicked
    def sample_pos_items_for_u(self, u, num):
        # sample num pos items for u-th user
        pos_items = self.train_items[u]
        n_pos_items = len(pos_items)
        pos_batch = []
        while True:
            if len(pos_batch) == num:
                break
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]

            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        return pos_batch

    # choose num items that u does not clicked
    def sample_neg_items_for_u(self, u, num):
        # sample num neg items for u-th user
        neg_items = []
        while True:
            if len(neg_items) == num:
                break
            neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
            if neg_id not in self.train_items[u] and neg_id not in neg_items:
                neg_items.append(neg_id)
        return neg_items
    
    def sample(self, mode='train'):
        if mode == 'train':
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = self.exist_test_users[self.batch_size*self.index : self.batch_size*(self.index+1)]
            self.index += 1
            print("    current batch: ", self.index)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += self.sample_pos_items_for_u(u, 1)
            neg_items += self.sample_neg_items_for_u(u, 1)

        test_pos_items, test_neg_items = [], []
        for u in users:
            test_pos_items += self.sample_pos_items_for_u(u, 1)
            test_neg_items += self.sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items, test_pos_items, test_neg_items
    
        


        
    

