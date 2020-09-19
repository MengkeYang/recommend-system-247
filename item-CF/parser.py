import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Item-CF Parser.")
    parser.add_argument('--train_path', nargs='?', default='',
                        help='Path of training data')
    parser.add_argument('--test_path', nargs='?', default='',
                        help='Path of testing data')
    parser.add_argument('--phase', type=int, default=6,
                        help='Current phase')

    return parser.parse_args()
