import os
import pandas as pd
import scipy.sparse as ssp
import pickle
from data_utils import *
from builder import PandasGraphBuilder
import dgl
import numpy as np
from numpy import genfromtxt
import dgl.function as fn
from dgl import DGLGraph
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import pickle


curr_phase = 6
directory = "/home/cocoa/247Proj/"
output_path = "users.dat"

click_data = genfromtxt(
    directory+f'/underexpose_train/underexpose_train_click-{0}.csv', delimiter=',', dtype=np.float32)
for i in range(1, curr_phase+1):
    click_data = np.concatenate((click_data, genfromtxt(
        directory+f'/underexpose_train/underexpose_train_click-{i}.csv', delimiter=',', dtype=np.float32)))
timestamps=torch.Tensor(click_data[:,2])
click_data = click_data[:, :2]
users = click_data[:, 0]
items = click_data[:, 1]
total_vertices = len(np.unique(users))+len(np.unique(items))
uid2vid = {}
vid2uid = {}
iid2vid = {}
vid2iid = {}
inc = 0
for user in np.unique(users):
    uid2vid[user] = inc
    vid2uid[inc] = user
    inc += 1
inc = 0
for item in np.unique(items):
    iid2vid[item] = inc
    vid2iid[inc] = item
    inc += 1
assert((len(iid2vid)+len(uid2vid)) == total_vertices)

src = list(map(lambda x: uid2vid[x], users))
dst = list(map(lambda x: iid2vid[x], items))

click_graph = dgl.bipartite(list(zip(src,dst)), 'user', 'ui', 'item')
click_graph.edges['ui'].data['timestamp']=timestamps
click_graph.edges['ui'].data['rating']=torch.ones(click_graph.number_of_edges())
clicked_graph = dgl.bipartite(list(zip(dst,src)), 'item', 'iu', 'user')
clicked_graph.edges['iu'].data['timestamp']=timestamps
clicked_graph.edges['iu'].data['rating']=torch.ones(clicked_graph.number_of_edges())
g = dgl.hetero_from_relations({click_graph, clicked_graph})

with open(directory+"/underexpose_train/user_generate_feat.txt", "r")as f:
    lines = f.readlines()
    usr_feat = np.zeros((len(lines), 5))
    for i in range(len(lines)):
        if lines[i].split(",")[2] == "0":
            usr_feat[i][3] = 1
        if lines[i].split(",")[2] == "1":
            usr_feat[i][4] = 1
    del lines
fn = directory+"/underexpose_train/user_generate_feat.txt"
usr_data = genfromtxt(fn, delimiter=',', dtype=np.int16)
usr_feat[:, 0:2] = usr_data[:, 0:2]
usr_feat[:, 2] = usr_data[:, 3]
uid2feat = {}
for i in range(len(usr_feat)):
    uid2feat[int(usr_feat[i][0])] = usr_feat[i][1:]
del usr_data

usr_v_feat = np.array([])  # feature matrix
usr_mean_feat = np.sum(np.array(list(uid2feat.values())), axis=0)/len(uid2feat)
for i in range(g.number_of_src_nodes("user")):
    if vid2uid[i] in uid2feat:
        usr_v_feat = np.append(usr_v_feat, uid2feat[vid2uid[i]])
    else:
        usr_v_feat = np.append(usr_v_feat, usr_mean_feat)
usr_v_feat = usr_v_feat.reshape((-1, 4))
g.nodes['user'].data['age'] = linear_normalize(torch.Tensor(usr_v_feat[:,0]))
g.nodes['user'].data['city'] = linear_normalize(torch.Tensor(usr_v_feat[:,1]))
g.nodes['user'].data['male'] = linear_normalize(torch.Tensor(usr_v_feat[:,2]))
g.nodes['user'].data['female'] = linear_normalize(torch.Tensor(usr_v_feat[:,3]))


item_data = genfromtxt(
    directory+"/underexpose_train/underexpose_item_feat_clean.csv", delimiter=',', dtype=np.float32)
item_ids = item_data[:, 0].astype(np.int)
item_feats = item_data[:, 1:].astype(np.float16)
iid2feat = {}
for i in range(len(item_ids)):
    iid2feat[item_ids[i]] = item_feats[i]
del item_data
item_mean_feat = np.sum(
    np.array(list(iid2feat.values())), axis=0)/len(iid2feat)
itm_v_feat = np.zeros((g.number_of_src_nodes("item"), item_mean_feat.shape[0]))
for i in range(g.number_of_src_nodes("item")):
    if vid2iid[i] in iid2feat:
        itm_v_feat[i] = iid2feat[vid2iid[i]]
    else:
        itm_v_feat[i] = item_mean_feat
g.nodes['item'].data['feats'] = torch.Tensor(itm_v_feat)

train_indices, val_indices, test_indices = train_test_split_by_time(g, 'timestamp', 'ui', 'item')
train_g = build_train_graph(g, train_indices, 'user', 'item', 'ui', 'iu')
# Build the user-item sparse matrix for validation and test set.
val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'item', 'ui')

dataset = {
    'train-graph': train_g,
    'val-matrix': val_matrix,
    'test-matrix': test_matrix,
    'item-texts': {},
    'item-images': None,
    'user-type': 'user',
    'item-type': 'item',
    'user-to-item-type': 'ui',
    'item-to-user-type': 'iu',
    'timestamp-edge-column': 'timestamp'}

with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)
