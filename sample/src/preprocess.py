import random
from sklearn.model_selection import train_test_split
import pickle as pkl
import os
from argparse import ArgumentParser
import spacy
import torch

random.seed(1)


def preprocess(args):
    pos_file = args.pos_file
    neg_file = args.neg_file
    pkl_path = args.pkl_path

    nlp = spacy.load(spacy_model)

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

    devel_data = [(1, p) for p in pos_devel]
    devel_data.extend([(0, n) for n in neg_devel])

    test_data = [(1, p) for p in pos_test]
    test_data.extend([(0, n) for n in neg_test])

    # reshape data
    text_train = [t for _, t in train_data]
    label_train = [lab for lab, _ in train_data]

    text_devel = [t for _, t in devel_data]
    label_devel = [lab for lab, _ in devel_data]

    text_test = [t for _, t in test_data]
    label_test = [lab for lab, _ in test_data]

    # tokenize text
    tokenized_train = [nlp.tokenizer(t) for t in text_train]
    tokenized_devel = [nlp.tokenizer(t) for t in text_devel]
    tokenized_test = [nlp.tokenizer(t) for t in text_test]

    # vocab
    vocab = set()
    for t in tokenized_train:
        v = set(map(lambda x: x.lower_, s))
        vocab = vocab | v
    voc = sorted(vocab)
    vocab = {}
    for i, v in enumerate(voc):
        vocab[v] = i

    # BoW vector
    def get_bow_vector(text, vocab):
        vec = torch.zeros(len(vocab))
        for t in text:
            lower = t.lower_
            if lower in vocab.keys():
                vec[vocab[lower]] = 1
        return vec

    bow_train = torch.cat([get_bow_vector(t, vocab).unsqueeze(dim=0) for t in tokenized_train], dim=0)
    bow_devel = torch.cat([get_bow_vector(t, vocab).unsqueeze(dim=0) for t in tokenized_devel], dim=0)
    bow_test = torch.cat([get_bow_vector(t, vocab).unsqueeze(dim=0) for t in tokenized_test], dim=0)

    # save data
    with open(os.path.join(pkl_path, 'train.pkl'), 'wb') as f:
        pkl.dump((tokenized_train, bow_train, label_train), f)
    with open(os.path.join(pkl_path, 'devel.pkl'), 'wb') as f:
        pkl.dump((tokenized_devel, bow_devel, label_devel), f)
    with open(os.path.join(pkl_path, 'test.pkl'), 'wb') as f:
        pkl.dump((tokenized_test, bow_test, label_test), f)
    with open(os.path.join(pkl_path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(vocab, f)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('pos_file', help='pos file path')
    parser.add_argument('neg_file', help='neg file path')
    parser.add_argument('pkl_path', help='output path (outputs are pickle)')
    parser.add_argument('--spacy_model', type=str, default='en_core_web_sm')

    args = parser.parse_args()

    preprocess(args)