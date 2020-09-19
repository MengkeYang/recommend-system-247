import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Hitting Time Parser.")
    parser.add_argument('--train_path', nargs='?', default='',
                        help='Path of training data')
    parser.add_argument('--test_path', nargs='?', default='',
                        help='Path of testing data')
    parser.add_argument('--phase', type=int, default=6,
                        help='Current phase')
    parser.add_argument('--K', type=int, default=50,
                        help='Top k(s) recommend')
    parser.add_argument('--max_iter', type=int, default=7,
                        help='Max iteration of Markov chain')
    parser.add_argument('--bfs_users', type=int, default=1000,
                        help='Maximum users in the sub graph')
    parser.add_argument('--bfs_items', type=int, default=4000,
                        help='Maximum items in the sub graph')

    return parser.parse_args()
