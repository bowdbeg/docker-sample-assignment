import torch
import torch.nn as nn
import random
import spacy
import pickle as pkl
import os
from models import BinaryClassifier
from torch.utils.tensorboard import SummeryWriter
from glob import glob
from utils import EarlyStopper
from tqdm import tqdm
from collections import OrderedDict

# fix random seed
random.seed(1)


def train(args):
    spacy_model = args.spacy_model
    hidden_size = args.hidden_size
    tbpath = args.tbpath
    model_save_path = args.model_save_path
    model_load_path = args.model_load_path
    job_tag = args.job_tag
    pkl_path = args.data_path
    hidden_layer = args.hidden_layer
    non_linear = args.non_linear
    epochs = args.epoch
    batch_size = args.batch_size
    lr = args.lr

    stop_patience = args.stop_patience
    stop_threshold = args.stop_threshold
    stop_startup = args.stop_startup

    schedule_patience = args.schedule_patience
    schedule_factor = args.schedule_factor
    schedule_cooldown = args.schedule_cooldown
    schedule_threshold = args.schedule_threshold

    nlp = spacy.load(spacy_model)
    device = torch.device(args.device)

    # load data
    train_data_path = os.path.join(pkl_path, 'train.pkl')
    devel_data_path = os.path.join(pkl_path, 'devel.pkl')
    test_data_path = os.path.join(pkl_path, 'test.pkl')

    with open(train_data_path, 'rb') as f:
        tokenized_train, bow_train, label_train = pkl.load(f)
    with open(devel_data_path, 'rb') as f:
        tokenized_devel, bow_devel, label_devel = pkl.load(f)
    with open(test_data_path, 'rb') as f:
        tokenized_test, bow_test, label_test = pkl.load(f)

    vocab_size = bow_train.size(-1)

    # upload to device
    bow_train = bow_train.to(device)
    label_train = label_train.to(device)

    bow_devel = bow_devel.to(device)
    label_devel = label_devel.to(device)

    bow_test = bow_test.to(device)
    label_test = label_test.to(device)

    # construct model
    model = models.BinaryClassifier(vocab_size,
                                    hidden_size,
                                    hidden_layers=hidden_layer,
                                    non_linear=non_linear)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=schedule_factor,
        patience=schedule_patience,
        cooldown=schedule_cooldown,
        threshold=schedule_threshold)

    # early stopping
    stopper = EarlyStopper(model,
                           patience=stop_patience,
                           threshold=stop_threshold,
                           startup=stop_startup,
                           mode='min')

    # loss function
    loss_func = nn.BCELoss()

    # tensorboard
    if os.path.exists(os.path.join(tbpath, '{:04}'.format(0))):
        trial_num = max(
            list(
                map(lambda x: int(os.path.basename(x)),
                    glob(os.path.join(tbpath, '*'))))) + 1
    else:
        trial_num = 0
    tb = os.path.join(tbpath, '{:04}'.format(trial_num))
    writer = SummaryWriter(tb)

    # train
    trainset = torch.utils.data.TensorDataset(bow_train, label_train)
    develset = torch.utils.data.TensorDataset(bow_devel, label_devel)
    testset = torch.utils.data.TensorDataset(bow_test, label_test)

    iteration = 0
    for epoch in range(epochs):
        train_itr = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=True)
        losses = []

        with tqdm(train_itr) as pbar:
            pbar.set_description('{:04}/[Epoch {:3}]'.format(trial_num, epoch))
            for _, itr in enumerate(pbar):
                bow, label = itr
                model.train()
                optimizer.zero_grad()

                pred = model(bow)
                loss = loss_func(pred,label)
                
                loss.backward()
                optimizer.step()
                writer.add_scalar('train/loss_itr', loss, iteration)
                pbar.set_postfix(OrderedDict(loss='{:.4}'.format(loss.item())))
                losses.append(loss.item())
                iteration += 1
        
        # evaluation
        with torch.no_grad():
            devel_itr = torch.utils.data.DataLoader(develset)
                