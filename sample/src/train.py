import torch
import torch.nn as nn
import random
import spacy
import pickle as pkl
import os
from models import BinaryClassifier
from glob import glob
from utils import EarlyStopper,
from tqdm import tqdm
from collections import OrderedDict
from statistics import mean

# fix random seed
random.seed(1)


def train(args):
    spacy_model = args.spacy_model
    hidden_size = args.hidden_size
    tbpath = args.tbpath
    job_tag = args.job_tag
    pkl_path = args.data_path
    hidden_layer = args.hidden_layer
    non_linear = args.non_linear
    epochs = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    eval_test = args.eval_test
    model_load_path = args.model_load_path

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
    if not os.path.exists(tb):
        os.makedirs(tb)
    writer = torch.utils.tensorboard.SummaryWriter(tb)

    model_save_path = os.path.join(args.model_save_path, job_tag, trial_num)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # train
    trainset = torch.utils.data.TensorDataset(bow_train, label_train)
    develset = torch.utils.data.TensorDataset(bow_devel, label_devel)
    testset = torch.utils.data.TensorDataset(bow_test, label_test)

    iteration = 0
    best_f1_devel = 0.
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
                loss = loss_func(pred, label)

                loss.backward()
                optimizer.step()
                writer.add_scalar('train/loss_itr', loss, iteration)
                pbar.set_postfix(OrderedDict(loss='{:.4}'.format(loss.item())))
                losses.append(loss.item())
                iteration += 1

        loss_train = mean(losses)
        writer.add_argument('train/loss', loss_train, epoch)

        # evaluation
        with torch.no_grad():
            model.eval()

            print('Evaluate Development data...')
            devel_itr = torch.utils.data.DataLoader(develset)
            preds = []
            losses = []
            for itr in devel_itr:
                bow, label = itr
                pred = model(bow)
                preds.append(model.predict_from_output(pred))
                loss = loss_func(pred, label)
                losses.append(loss)
            print('finished')
            preds = torch.cat(preds, dim=0)
            loss_devel = mean(losses)
            precision_devel, recall_devel, f1_devel = calc_score(
                preds, label_devel)
            writer.add_scalar('devel/precision', precision_devel, epoch)
            writer.add_scalar('devel/recall', recall_devel, epoch)
            writer.add_scalar('devel/f1', f1_devel, epoch)

            if eval_test:
                print('Evaluate Test data...')
                test_itr = torch.utils.data.DataLoader(testset)
                preds = []
                losses = []
                for itr in test_itr:
                    bow, label = itr
                    pred = model(bow)
                    preds.append(model.predict_from_output(pred))
                    loss = loss_func(pred, label)
                    losses.append(loss)
                print('finished')
                preds = torch.cat(preds, dim=0)
                loss_test = mean(losses)
                precision_test, recall_test, f1_test = calc_score(
                    preds, label_test)
                writer.add_scalar('test/precision', precision_test, epoch)
                writer.add_scalar('test/recall', recall_test, epoch)
                writer.add_scalar('test/f1', f1_test, epoch)

        # learning rate scheduler
        scheduler.step(loss_devel)

        # when get best score
        if best_f1_devel <= f1_devel:
            best_devel_f1 = devel_f1
            best_weight = net.state_dict()
            best_epoch = epoch

        if epoch % 5 == 4:
            print('Saving best model (epoch {})...'.format(best_epoch))
            torch.save(
                best_weight,
                os.path.join(model_save_path, '{:04}.pth'.format(best_epoch)))
            print('finished')
            if best_epoch != epoch:
                print('Saving checkpoint model (epoch {})...'.format(epoch))
                torch.save(
                    net.state_dict(),
                    os.path.join(model_save_path,
                                 'checkpoint_{:04}.pth'.format(epoch)))
                print('finished')

        # early stopping
        if stopper.step(loss_devel):
            break

    # save best model
    print('Saving final best model ...')
    torch.save(
        stopper.best_param,
        os.path.join(model_save_path,
                     'best_{:04}.pth'.format(stopper.best_epoch)))