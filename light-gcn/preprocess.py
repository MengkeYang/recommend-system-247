import argparse
import pandas as pd
from tqdm import trange
from utility.helper import ensureDir


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess KDD Data.")
    parser.add_argument('--train_path', nargs='?', default='',
                        help='Path of training data')
    parser.add_argument('--test_path', nargs='?', default='',
                        help='Path of testing data')
    parser.add_argument('--data_path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--name', nargs='?', default='kdd',
                        help='Name of the dataset')
    parser.add_argument('--phase', type=int, default=6,
                        help='Current phase')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='Train test split ratio')
    return parser.parse_args()


args = parse_args()

now_phase = args.phase
train_ratio = args.ratio
data_path = '%s%s/' % (args.data_path, args.name)
ensureDir(data_path)
whole_click = pd.DataFrame()

# Load data
for c in trange(now_phase + 1):
    click_train = pd.read_csv(args.train_path + 'underexpose_train_click-{}.csv'.format(c), header=None,
                              names=['user_id', 'item_id', 'time'])
    click_test = pd.read_csv(args.test_path + 'underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c),
                             header=None,
                             names=['user_id', 'item_id', 'time'])

    whole_click = whole_click.append(click_train).drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    whole_click = whole_click.append(click_test).drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')

whole_click.sort_values('time', inplace=True)

unique_user_id = list(whole_click['user_id'].unique())
unique_item_id = list(whole_click['item_id'].unique())
pd.DataFrame(list(zip(unique_user_id, range(len(unique_user_id)))), columns=['org_id', 'remap_id']).to_csv(
    data_path + 'user_list.txt', index=False, sep=' ')
pd.DataFrame(list(zip(unique_item_id, range(len(unique_item_id)))), columns=['org_id', 'remap_id']).to_csv(
    data_path + 'item_list.txt', index=False, sep=' ')

user_id_map = dict(zip(unique_user_id, range(len(unique_user_id))))
item_id_map = dict(zip(unique_item_id, range(len(unique_item_id))))
whole_click['user_id'] = whole_click['user_id'].map(user_id_map)
whole_click['item_id'] = whole_click['item_id'].map(item_id_map)

f_train = open(data_path + 'train.txt', 'w')
f_test = open(data_path + 'test.txt', 'w')

for user_id in trange(len(user_id_map)):
    item_ids = whole_click[whole_click['user_id'] == user_id]['item_id'].astype(str).values.tolist()
    size = int(len(item_ids) * train_ratio)
    if size == 0:
        f_train.write(str(user_id) + ' ' + item_ids[0] + '\n')
        f_test.write(str(user_id) + ' ' + item_ids[0] + '\n')
    else:
        f_train.write(' '.join([str(user_id)] + item_ids[:size]) + '\n')
        f_test.write(' '.join([str(user_id)] + item_ids[size:]) + '\n')

f_train.flush()
f_test.flush()

f_train.close()
f_test.close()

print('Data preprocess finished.')
