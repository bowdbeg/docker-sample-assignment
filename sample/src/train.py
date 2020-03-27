import torch
import torch.nn as nn
import random

# fix random seed
random.seed(1)

def train(args):
    spacy_model = args.spacy_model
    hidden_size = args.hidden_size
    tbpath = args.tbpath
    model_save_path = args.model_save_path
    model_load_path = args.model_load_path
    job_tag = args.job_tag

    nlp = spacy.load(spacy_model)
    device = torch.device('cuda:0')

    