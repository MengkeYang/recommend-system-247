from utility.parser import parse_args
from utility.load_data import Data
import multiprocessing
import heapq
from tqdm import tqdm

cores = multiprocessing.cpu_count() // 2
args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path, batch_size=args.batch_size)
ITEM_NUM = data_generator.n_items
BATCH_SIZE = args.batch_size


def ranklist_by_heapq(test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(50, item_score, key=item_score.get)
    return K_max_item_score


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    training_items = data_generator.train_items[u]

    all_items = set(range(ITEM_NUM))
    test_items = list(all_items - set(training_items))

    r = ranklist_by_heapq(test_items, rating, Ks)
    return {u: r}


def test(model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = []

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0

    for u_batch_id in tqdm(range(n_user_batchs)):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]
        # all-item test
        item_batch = range(ITEM_NUM)

        u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                        item_batch,
                                                        [],
                                                        drop_flag=False)
        rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)
        result.append(batch_result)

    assert count == n_test_users
    pool.close()
    return result
