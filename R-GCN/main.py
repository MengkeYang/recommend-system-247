from model import HeteroRGCN
from loss import bpr_loss
from data import Data
from batch_test import test
import pandas as pd
import torch
from tqdm import tqdm 
import argparse
from time import gmtime,strftime

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', default = 6, type = int,help='current phase of the competition')
    parser.add_argument('--train_data_dir', default='./data/underexpose_train')
    parser.add_argument('--test_data_dir', default='./data/underexpose_test')
    parser.add_argument('--map_data_dir', default='./data/mydata')
    parser.add_argument('--feature_data_dir', default='./data/mydata')
    parser.add_argument('--weights_path', nargs='?', default='./data/weight/')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--patience', default = 5, type = int)
    parser.add_argument('--in_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--out_size', type=int, default=32)
    parser.add_argument('--self_loop', action='store_true')
    parser.add_argument('--drop_out', type=float, default=0.0)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wdc', type=float, default=0.0)
    parser.add_argument('--Ks', type=int, default=50,
                        help='Output sizes of every layer')

    return parser.parse_args()

args = parse_args()

#load item_feat
item_feat = pd.read_csv(args.feature_data_dir + '/underexpose_item_feat.csv', header=None)
col_name_list = ["item_id"] + ["txt_vec_" + str(x) for x in range(128)] + ["img_vec_" + str(x) for x in range(128)]
item_feat.columns = col_name_list
item_feat['txt_vec_0'] = item_feat['txt_vec_0'].str.replace('[', '')
item_feat['txt_vec_127'] = item_feat['txt_vec_127'].str.replace(']', '')
item_feat['img_vec_0'] = item_feat['img_vec_0'].str.replace('[', '')
item_feat['img_vec_127'] = item_feat['img_vec_127'].str.replace(']', '')

item_mapped_id = pd.read_csv(args.map_data_dir + '/item_list.txt',sep=' ')

item_mapped_df = pd.merge(left=item_mapped_id, right=item_feat, left_on='org_id', right_on='item_id')
item_mapped_df = item_mapped_df.drop(['org_id','item_id'],axis=1)
item_mapped_df = item_mapped_df.rename(columns={'remap_id':'mapped_item_id'})

#load user_feat
#user_feat = pd.read_csv(args.feature_data_dir  + '/underexpose_user_feat.csv', header=None, names=['user_id','user_age_level','user_gender','user_city_level'])
#user_mapped_id = pd.read_csv(args.map_data_dir + '/user_list.txt',sep=' ')

#load train & test data
#tain on previous phase to avoid data leakage
now_phase = args.phase
all_click_train = pd.DataFrame()
all_click_test = pd.DataFrame()
all_click_qtime = pd.DataFrame()
whole_click = pd.DataFrame()
for c in range(now_phase + 1):
    print('Loading data: phase:', c)
    test_path = args.test_data_dir+'/underexpose_test_click-{}'.format(c)
    click_train = pd.read_csv(args.train_data_dir + '/underexpose_train_click-{}.csv'.format(c), header=None,
                              names=['user_id', 'item_id', 'time'])
    click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c), header=None,
                             names=['user_id', 'item_id', 'time'])
    click_qtime = pd.read_csv(test_path + '/underexpose_test_qtime-{}.csv'.format(c), header=None,
                             names=['user_id', 'time'])
    all_click_train = all_click_train.append(click_train)
    all_click_test = all_click_test.append(click_test)
    all_click_qtime = all_click_qtime.append(click_qtime)

    all_click_train = all_click_train.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    all_click_test = all_click_test.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')

    whole_click = all_click_train.append(all_click_test)
    whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    whole_click = whole_click.sort_values('time')

user_mapped_id = {org_id: remap_id for remap_id, org_id in enumerate(whole_click['user_id'].unique())}
user_mapped_id_df_ =  pd.DataFrame(user_mapped_id.items(), columns=['org_id', 'remap_id'])
data = Data(whole_click,all_click_test, all_click_qtime,user_mapped_id_df_,item_mapped_id ,args.batch_size)

item_mapped_df_subset = item_mapped_df[item_mapped_df['mapped_item_id'].isin(data.items)]

model = HeteroRGCN(data.G,item_mapped_df_subset ,args.in_size, args.hidden_size, args.out_size, args.bias, args.self_loop, args.drop_out)

opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdc)

for epoch in range(args.epoch):
    model.train()
    logits = model(data.G)

    loss, mf_loss, emb_loss = 0., 0., 0.
    n_batch = data.n_train // args.batch_size + 1

    for idx in tqdm(range(n_batch),desc='epoch '+str(epoch)):
        users, pos_items, neg_items = data.sample()
        batch_mf_loss, batch_emb_loss = bpr_loss(logits['user'][users], logits['item'][pos_items], logits['item'][neg_items])
        loss = loss + batch_mf_loss + batch_emb_loss
        mf_loss += batch_mf_loss.item()
        emb_loss += batch_emb_loss.item()

    print("Computing...")
    opt.zero_grad()
    loss.backward()
    opt.step()

    print("Epoch {}: loss {}, emb_loss {}, mf_loss {}".format(epoch, loss.item(), emb_loss, mf_loss))

    if epoch%100 == 0:
        time_now = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        weight_file_name = time_now+'weight.pt'
        weight_file_path = args.weights_path+weight_file_name
        torch.save(model.state_dict(),weight_file_path)
    if loss.item() < 0.01:
        break


model.eval()
logits = model(data.G)
print("Generating recommendation...")
test(data, logits, args.Ks)

