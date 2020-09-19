import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
import  numpy as np
from    NGCF import NGCF
from tqdm import tqdm


class Meta(nn.Module):
    def __init__(self, args, n_user, n_item, norm_adj):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.task_num = args.batch_size
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = NGCF(n_user, n_item, norm_adj, args)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def forward(self, users, pos_items, neg_items, test_pos_items, test_neg_items):

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        # 1. run the i-th task and compute loss for k=0
        u_g_embedding, pos_i_g_embedding, neg_i_g_embedding = self.net(users, pos_items, neg_items, vars=None)
        loss, _, _ = self.net.create_bpr_loss(u_g_embedding, pos_i_g_embedding, neg_i_g_embedding)
        grad = torch.autograd.grad(loss, self.net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
        
        # this is the loss and accuracy before first update
        with torch.no_grad():
            u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q = self.net(users, test_pos_items, test_neg_items, self.net.parameters())
            loss_q, _, _ = self.net.create_bpr_loss(u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q)
            losses_q[0] += loss_q

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q = self.net(users, test_pos_items, test_neg_items, fast_weights)
            loss_q, _, _ = self.net.create_bpr_loss(u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q)
            losses_q[1] += loss_q

        for k in range(1, self.update_step):
            # 1. run the i-th task and compute loss for k=1~K-1
            u_g_embedding, pos_i_g_embedding, neg_i_g_embedding = self.net(users, pos_items, neg_items, fast_weights)
            loss, _, _ = self.net.create_bpr_loss(u_g_embedding, pos_i_g_embedding, neg_i_g_embedding)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, self.net.parameters())
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q = self.net(users, test_pos_items, test_neg_items, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q, _, _ = self.net.create_bpr_loss(u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q)
            losses_q[k + 1] += loss_q

        # end of all tasks
        loss_q = losses_q[-1]

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        # average testing acc on batch tasks for different uodate step
        return [one.tolist() for one in losses_q]
    

    def finetunning(self, users, pos_items, neg_items, test_pos_items, test_neg_items):

        losses_q = [0 for _ in range(self.update_step_test + 1)]  # losses_q[i] is the loss on step i

        # 1. run the i-th task and compute loss for k=0
        u_g_embedding, pos_i_g_embedding, neg_i_g_embedding = self.net(users, pos_items, neg_items, vars=None)
        loss, _, _ = self.net.create_bpr_loss(u_g_embedding, pos_i_g_embedding, neg_i_g_embedding)
        grad = torch.autograd.grad(loss, self.net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
        
        # this is the loss and accuracy before first update
        with torch.no_grad():
            u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q = self.net(users, test_pos_items, test_neg_items, self.net.parameters())
            loss_q, _, _ = self.net.create_bpr_loss(u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q)
            losses_q[0] += loss_q

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q = self.net(users, test_pos_items, test_neg_items, fast_weights)
            loss_q, _, _ = self.net.create_bpr_loss(u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q)
            losses_q[1] += loss_q
    
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            u_g_embedding, pos_i_g_embedding, neg_i_g_embedding = self.net(users, pos_items, neg_items, fast_weights)
            loss, _, _ = self.net.create_bpr_loss(u_g_embedding, pos_i_g_embedding, neg_i_g_embedding)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, self.net.parameters())
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q = self.net(users, test_pos_items, test_neg_items, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q, _, _ = self.net.create_bpr_loss(u_g_embedding_q, pos_i_g_embedding_q, neg_i_g_embedding_q)
            losses_q[k + 1] += loss_q

        return [one.tolist() for one in losses_q]
