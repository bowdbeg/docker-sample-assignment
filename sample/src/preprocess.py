import random
from sklearn.model_selection import train_test_split
import pickle as pkl
import os
from argparse import ArgumentParser

random.seed(1)


def preprocess(args):
    pos_file = args.pos_file
    neg_file = args.neg_file
    pkl_path = args.pkl_path

    with open(pos_file, encoding='utf-8', errors='ignore') as f:
        neg = f.read().strip().split('\n')

    with open(neg_file, encoding='utf-8', errors='ignore') as f:
        pos = f.read().strip().split('\n')

    pos_train, pos_tmp, neg_train, neg_tmp = train_test_split(pos,
                                                              neg,
                                                              train_size=0.6,
                                                              random_state=0)
    pos_devel, pos_test, neg_devel, neg_test = train_test_split(pos_tmp,
                                                                neg_tmp,
                                                                train_size=0.5,
                                                                random_state=0)

    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    train_data = [(1, p) for p in pos_train]
    train_data.extend([(0, n) for n in neg_train])
    with open(os.path.join(pkl_path, 'train.pkl'), 'wb') as f:
        pkl.dump(train_data, f)

    devel_data = [(1, p) for p in pos_devel]
    devel_data.extend([(0, n) for n in neg_devel])
    with open(os.path.join(pkl_path, 'devel.pkl'), 'wb') as f:
        pkl.dump(devel_data, f)

    test_data = [(1, p) for p in pos_test]
    test_data.extend([(0, n) for n in neg_test])
    with open(os.path.join(pkl_path, 'test.pkl'), 'wb') as f:
        pkl.dump(test_data, f)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('pos_file', help='pos file path')
    parser.add_argument('neg_file', help='neg file path')
    parser.add_argument('pkl_path', help='output path (outputs are pickle)')

    args = parser.parse_args()

    preprocess(args)