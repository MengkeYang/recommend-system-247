from collections import defaultdict
import dgl
import pandas as pd
import numpy as np
import random


class Data():
    def __init__(self, all_click_train,all_click_test,click_qtime, user_mapped_id,item_mapped_id,batch_size=128):

        self.mapped_train_df = self.map_id_df(all_click_train,user_mapped_id,item_mapped_id)
        self.mapped_test_df = self.map_id_df(all_click_test,user_mapped_id,item_mapped_id)

        self.mapped_qtime_df = pd.merge(left=click_qtime, right=user_mapped_id, left_on='user_id', right_on='org_id')
        self.mapped_qtime_df =  self.mapped_qtime_df.drop(['org_id'], axis=1)
        self.mapped_qtime_df =  self.mapped_qtime_df.rename(columns={'remap_id': 'mapped_user_id'})
        self.qtime_users = self.mapped_qtime_df['mapped_user_id'].tolist()

        self.train_items = self.get_users_items_dict(self.mapped_train_df)
        self.test_items = self.get_users_items_dict(self.mapped_test_df)

        self.train_edge_dict = self.get_edge_list(self.mapped_train_df)
        self.test_edge_dict = self.get_edge_list(self.mapped_test_df)

        self.n_train = len(self.train_edge_dict[('user','ui','item')])
        self.n_test = len(self.test_edge_dict[('user','ui','item')])

        self.mapped_userId_to_org_userId = dict(zip(user_mapped_id.remap_id,user_mapped_id.org_id))
        self.mapped_itemId_to_org_itemId = dict(zip(item_mapped_id.remap_id, item_mapped_id.org_id))

        self.G = dgl.heterograph(self.train_edge_dict)

        self.n_items = self.G.number_of_nodes('item')
        self.n_users = self.G.number_of_nodes('user')
        self.users = self.G.nodes('user').detach().cpu().numpy().tolist()
        self.items = self.G.nodes('item').detach().cpu().numpy().tolist()

        self.batch_size = batch_size


    def map_id_df(self,df,user_mapped_id_df,item_mapped_id_df):
        mapped_df = pd.merge(left=df, right=user_mapped_id_df, left_on='user_id', right_on='org_id')
        mapped_df = mapped_df.drop(['org_id'], axis=1)
        mapped_df = mapped_df.rename(columns={'remap_id': 'mapped_user_id'})
        mapped_df = pd.merge(left=mapped_df, right=item_mapped_id_df, left_on='item_id', right_on='org_id')
        mapped_df = mapped_df.drop(['org_id'], axis=1)
        mapped_df = mapped_df.rename(columns={'remap_id': 'mapped_item_id'})
        mapped_df = mapped_df.sort_values(['mapped_user_id', 'time'])
        return mapped_df

    def get_users_items_dict(self,df):
        users_items_dict = defaultdict(list)
        all_users = df['mapped_user_id'].unique()
        for uid in all_users:
            users_items_dict[uid] = df[df['mapped_user_id'] == uid]['mapped_item_id'].values.tolist()
        return users_items_dict


    def get_edge_list(self,df):
        edge_dict = defaultdict(list)
        for index, row in df.iterrows():
            edge_dict[('user', 'ui', 'item')].append((int(row['mapped_user_id']), int(row['mapped_item_id'])))
            edge_dict[('item', 'iu', 'user')].append((int(row['mapped_item_id']), int(row['mapped_user_id'])))
        return edge_dict


    def sample(self):
        if self.batch_size <= self.n_users:
            #users = random.sample(self.users, self.batch_size)
            all_users_train = list(self.mapped_train_df['mapped_user_id'].unique())
            users = random.sample(all_users_train, self.batch_size)
        else:
            users = [random.choice(self.users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                if n_pos_items == 0: print(u)
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items


        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
