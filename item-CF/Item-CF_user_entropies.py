import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math
from parser import parse_args

args = parse_args()


def get_sim_item(all_click, user_col, item_col, time_col, use_iuf=False, use_time=False):
    user_item = all_click.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item[user_col], user_item[item_col]))

    user_time = all_click.groupby(user_col)[time_col].agg(list).reset_index()
    user_time_dict = dict(zip(user_time[user_col], user_time[time_col]))

    sim_item = {}
    item_cnt = defaultdict(int)
    for user, items in tqdm(user_item_dict.items()):
        for ixi, i in enumerate(items):
            item_cnt[i] += 1
            sim_item.setdefault(i, {})

            for ixj, j in enumerate(items):
                if i == j:
                    continue
                sim_item[i].setdefault(j, 0)

                t1 = user_time_dict[user][ixi]
                t2 = user_time_dict[user][ixj]

                if not use_iuf:
                    sim_item[i][j] += 1
                else:
                    if not use_time:
                        sim_item[i][j] += 1 / math.log(1 + len(items))
                    else:
                        flag = True if ixi > ixj else False
                        num = max(ixi - ixj, ixj - ixi) - 1
                        t = max(t1 - t2, t2 - t1) * 10000
                        indicator = 1.0 if flag else 1.0
                        sim_item[i][j] += 1.0 * indicator * (0.8 ** num) * (1 - t) / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()
    for i, related_items in tqdm(sim_item.items()):
        for j, sim in related_items.items():
            # sim_item_corr[i][j] = sim / math.sqrt(item_cnt[i] * item_cnt[j])
            sim_item_corr[i][j] = sim / (item_cnt[i] * item_cnt[j]) ** 0.2

    return sim_item_corr, user_item_dict


def recommend(item_sim_list, user_item_dict, user_id, topk, item_num):
    user_items = user_item_dict[user_id]
    user_items = user_items[::-1]
    rank = {}
    for ixi, i in enumerate(user_items):
        for j, sim in sorted(item_sim_list[i].items(), key=lambda x: x[1], reverse=True)[: topk]:
            if j not in user_items:
                rank.setdefault(j, 0)
                rank[j] += sim * (0.7 ** ixi)
    return sorted(rank.items(), key=lambda x: x[1], reverse=True)[: item_num]


def get_predict(rec_df, pred_col, top_50_clicks):
    top_50_clicks = [int(t) for t in top_50_clicks.split(',')]
    scores = [-1 * (i + 1) for i in range(0, len(top_50_clicks))]
    ids = list(rec_df['user_id'].unique())

    fill_df = pd.DataFrame(ids * len(top_50_clicks), columns=['user_id'])
    fill_df.sort_values('user_id', inplace=True)
    fill_df['item_id'] = top_50_clicks * len(ids)
    fill_df[pred_col] = scores * len(ids)
    rec_df = rec_df.append(fill_df)
    rec_df.sort_values(pred_col, ascending=False, inplace=True)
    rec_df = rec_df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    rec_df['rank'] = rec_df.groupby('user_id')[pred_col].rank(method='first', ascending=False)
    rec_df = rec_df[rec_df['rank'] <= 50]
    rec_df = rec_df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',',
                                                                                                           expand=True).reset_index()

    return rec_df


current_phase = args.phase
train_path = args.train_path
test_path = args.test_path
rec_items = []

whole_click = pd.DataFrame()
for phase in range(current_phase + 1):
    print("phase: ", phase)
    train_click = pd.read_csv(train_path + 'underexpose_train_click-{}.csv'.format(phase), header=None,
                              names=['user_id', 'item_id', 'time'])
    test_click = pd.read_csv(test_path + 'underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(phase, phase),
                             header=None, names=['user_id', 'item_id', 'time'])
    test_users = pd.read_csv(test_path + 'underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(phase, phase),
                             header=None, names=['user_id', 'time'])

    all_click = train_click.append(test_click)
    whole_click = whole_click.append(all_click)
    whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    whole_click = whole_click.sort_values('time')

    item_sim_list, user_item = get_sim_item(whole_click, 'user_id', 'item_id', 'time', use_iuf=True, use_time=True)

    for i in tqdm(test_users['user_id'].unique()):
        rank_items = recommend(item_sim_list, user_item, i, 500, 50)
        for j in rank_items:
            rec_items.append([i, j[0], j[1]])

# find most popular items for cold-start users
top_50_clicks = whole_click['item_id'].value_counts().index[:50].values
top_50_clicks = ','.join([str(i) for i in top_50_clicks])

rec_df = pd.DataFrame(rec_items, columns=['user_id', 'item_id', 'sim'])
result = get_predict(rec_df, 'sim', top_50_clicks)
result.to_csv('pred.csv', index=False, header=None)
