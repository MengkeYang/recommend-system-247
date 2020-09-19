import  torch, os
import  numpy as np
# data provider
from    kddNShot import kddNShot
# parameter parser
import  argparse
# MAML nn.Module
from    kdd_meta import Meta
# generate ratings
from    batch_test import Test

def main(args):
    torch.manual_seed(222)
    # torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    print(args)
    # get training set (pos_neg_samples) and testing set (pos_neg_samples) of batch of tasks
    db_train = kddNShot(args.data_path, args.batch_size)

    # device = torch.device('cuda')
    ngcf_maml = Meta(args, db_train.data_generator.n_users, db_train.data_generator.n_items, db_train.norm_adj)# .to(device)

    # meta learning epoch: given a batch of tasks, 
    # update initial model parameters based on N-step trained parameters and testing set loss
    for step in range(args.epoch):
        # the batch of args.task_num tasks
        # training set and testing set of batch tasks
        users, pos_items, neg_items, test_pos_items, test_neg_items = db_train.next()
        # average testing loss
        loss_val = ngcf_maml(users, pos_items, neg_items, test_pos_items, test_neg_items)

        if step % 50 == 0:
            print('step:', step, '\taverage testing loss:', loss_val)
    
    for _ in range(len(db_train.data_generator.exist_test_users) // args.batch_size + 1):
        users, pos_items, neg_items, test_pos_items, test_neg_items = db_train.get_test()
        loss_val = ngcf_maml.finetunning(users, pos_items, neg_items, test_pos_items, test_neg_items)
        print('batch average testing loss on target users:', loss_val)


    users_to_test = db_train.data_generator.exist_test_users
    tester = Test(args, db_train.data_generator)
    ret = tester.test(ngcf_maml.net, users_to_test, drop_flag=False)
    db_train.data_generator.generate_answer(ret)

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MAML x NGCF.")
    # meta learning rate
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0001)
    # specific task learning rate
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.0005)
    # path to load data
    parser.add_argument('--data_path', nargs='?', default='../underexpose_train/', help='Input data path.')
    # number of tasks in one epoch of MAML
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    # epochs in a specific task when inner-loop training 
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    # epochs in a specific task when inner-loop testing
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    # NGCF input layer size
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    # NGCF node dropout
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]', help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    # NGCF message dropout
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]', help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    # NGCF size of layers
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]', help='Output sizes of every layer')
    # NGCF regularizations
    parser.add_argument('--regs', nargs='?', default='[1e-5]', help='Regularizations.')
    # MAML epochs
    parser.add_argument('--epoch', type=int, default=3, help='Number of epoch.')
    args = parser.parse_args()
    main(args)
