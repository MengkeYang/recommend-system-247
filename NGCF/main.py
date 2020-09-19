import torch
import torch.optim as optim
from NGCF import NGCF
# data will be loaded in batch_test
from utility.batch_test import *
from time import time
from tqdm import tqdm 


if __name__ == '__main__':
    args.device = torch.device('cuda:' + str(args.gpu_id))
    norm_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in tqdm(range(n_batch),desc='epoch '+str(epoch)):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
            epoch, time() - t1, loss, mf_loss, emb_loss)
        print(perf_str)

    users_to_test = data_generator.exist_users
    ret = test(model, users_to_test, drop_flag=False)
    data_generator.generate_answer(ret)