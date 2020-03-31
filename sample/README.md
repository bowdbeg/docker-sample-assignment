# TTI-COIN Sample Binary Classifier

Binary Classifier for [sentence polarity dataset v1.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz) of [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/).
This code was prepared for TTI-COIN's assignment.

## Requirements

Python Version 3.7.0

- torch>=1.4.0
- sklearn
- tqdm
- spacy
- tensorboard

## Quick Start

Run preprocess and [src/train.py](src/train.py).

```sh
sh preprocess.sh
python src/train.py --hidden_size 512 --hidden_layer 5 --epoch 100 --batch_size 32--lr 1e-2 --eval_test --device cuda:0
```

## Preprocess

- Download dataset from [sentence polarity datasetv1.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz) of [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/) and extract data.

```sh
wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
tar xvf rt-polaritydata.tar.gz
```

- Preprocessing data
  - Split dataset to train, devel and test set.
  - Prepare feature vector of sentences. (Bag of Words)

```sh
mv rt-polaritydata data
python src/preprocess.py data/rt-polarity.pos data/rt-polarity.neg pkl
```

## Training

Train MLP model for binary classification.

```sh
python src/train.py (options)
```

Training Option:

- --tbpath
  - Tensorboard path (default: 'runs/')
  - Tensorboard file will be saved to (tbpath)/(job_tag)/(trial_number)
- --job_tag
  - Job name of training (default: 'sample')
- --model_save_path
  - Path to save model
  - Model will be saved to (model_save_path)/(job_tag)/(trial_num)
    - tmp_(epoch).pkl: Temporaly good model besed on development F1 score
    - checkpoint_(epoch).pkl: Checkpoint model saved every 5 epochs
    - best_(epoch).pkl: Best model besed on Early Stopping
- --model_load_path
  - Pretrained model's weight file. (default: None)
- --epoch
  - Training epoch. (default: 1)
- --batch_size
  - Batch size of mini-batch learning. (default: 1)
- --eval_batch_size
  - Batch size of evaluation (default: 2048)
  - i.e. Batch size when gradient calculation is disabled
- --lr
  - Learning rate of optimizer (Adam). (default: 1e-2)
- --eval_test
  - Switch to run evaluation of test in each epoch. (default: False)
- --device
  - Select training device (default: 'cpu')
  - {'cpu', 'cuda', 'cuda:0', 'cuda:1', ...}

Model Parameters:

- --hidden_size
  - Hidden layer's dimension (default: 512)
- --hidden_layer 
  - A number of hidden layer (default: 1)
- --dropout_in
  - Dropout ratio of input layer (default: 0.0)
- --dropout_out
  - Dropout ratio of output layer. (default: 0.0)
- --non_linear
  - Select non-linear layer. (default: 'relu')
  - Choices: {'relu', 'tanh'}

Parameters for [Early Stopping](https://en.wikipedia.org/wiki/Early_stopping):

- --stop_patience
  - When development losses are not change (stop_patience) epochs, stop training.
- --stop_threshold
  - Margin of judgement of Early Stopping
  - If -(stop_threshold) < change of develpment loss < (stop_threshold)
- --stop_startup
  - Training will not stop for (stop_startup) epochs
- --no_early_stopping
  - Switch to run without Early Stopping. (default: False)

Parameters for [scheduler](https://pytorch.org/docs/stable/optim.html?highlight=pla#torch.optim.lr_scheduler.ReduceLROnPlateau) of learning rate:

- --schedule_patience
- --schedule_factor
- --schedule_cooldown
- --schedule_threshold
