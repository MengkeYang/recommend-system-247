from algorithm import *
from models import *
from parser import parse_args

args = parse_args()


def bfs(query_id, data: pd.DataFrame, max_user, max_item):
    user_ids = [query_id]
    item_ids, visited_user_ids, visited_item_ids = [], [], []
    all_user_ids, all_item_ids = {query_id}, set()
    while len(all_item_ids) < max_item or len(all_user_ids) < max_user:
        if len(item_ids) + len(visited_item_ids) < max_item and len(user_ids) > 0:
            query_id = user_ids[0]
            visited_user_ids.append(query_id)
            user_ids = user_ids[1:]
            link_ids = data[data['user_id'] == query_id].drop_duplicates(subset=['item_id'])
            link_ids = list(np.array(link_ids[~link_ids['item_id'].isin(all_item_ids)]['item_id']))
            all_item_ids.update(link_ids)
            item_ids += link_ids

        if len(user_ids) + len(visited_user_ids) < max_user and len(item_ids) > 0:
            query_id = item_ids[0]
            visited_item_ids.append(query_id)
            item_ids = item_ids[1:]
            link_ids = data[data['item_id'] == query_id].drop_duplicates(subset=['user_id'])
            link_ids = list(np.array(link_ids[~link_ids['user_id'].isin(all_user_ids)]['user_id']))
            all_user_ids.update(link_ids)
            user_ids += link_ids

        if len(user_ids) == 0 and len(item_ids) == 0:
            raise Exception('Insufficient links')

    user_ids = (visited_user_ids + user_ids)[:max_user]
    item_ids = (visited_item_ids + item_ids)[:max_item]

    return data[data['user_id'].isin(user_ids) & data['item_id'].isin(item_ids)]


def recommend(query_id, data: pd.DataFrame):
    user_ids = list(np.unique(data['user_id']))
    item_ids = list(np.unique(data['item_id']))
    user_id_map = dict(zip(user_ids, range(len(user_ids))))
    item_id_map = dict(zip(item_ids, range(len(item_ids))))
    data['user_id'] = data['user_id'].map(user_id_map)
    data['item_id'] = data['item_id'].map(item_id_map)
    item_id_map = dict(zip(range(len(item_ids)), item_ids))

    query_id = user_id_map[query_id]
    model = KDDModel(len(user_ids), len(item_ids), data)
    excluded_dict = {query_id: np.array(data[data['user_id'] == query_id]['item_id'])}
    return list(map(item_id_map.get,
                    (HittingTimeAlgorithm(model).run([query_id], excluded_dict, args.K, time_steps=args.max_iter))[0]))


now_phase = args.phase
whole_click = pd.DataFrame()
pred = open('pred.csv', 'w')
# Load data
for c in range(now_phase + 1):
    print('phase:', c)
    click_train = pd.read_csv(args.train_path + 'underexpose_train_click-{}.csv'.format(c), header=None,
                              names=['user_id', 'item_id', 'time'])
    click_test = pd.read_csv(args.test_path + 'underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c),
                             header=None,
                             names=['user_id', 'item_id', 'time'])

    whole_click = whole_click.append(click_train)
    whole_click = whole_click.append(click_test)
    whole_click.sort_values('time', ascending=False, inplace=True)
    whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'])

    for test_id in tqdm(click_test['user_id'].unique()):
        pred.write(' '.join(
            list(map(str, [test_id] + recommend(test_id, bfs(test_id, whole_click, args.bfs_users, args.bfs_items))))))
        pred.write('\n')

pred.close()
