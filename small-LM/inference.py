import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model_Vpass import GPT
from utils import *  # contains all of the helper methods

cfg = load_config("configs/config-8M.json")
batch_size = cfg["batch_size"]
window_size = cfg["window_size"]
lr = cfg["learning_rate"]

model_name = 'roneneldan/TinyStories'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT(cfg)
if torch.cuda.device_count() > 1:
    # if multiple gpus on single machine
    model = nn.DataParallel(model)
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

checkpoint = torch.load('models/model_8M_1003_185803.pt.tar')
model.load_state_dict(checkpoint['state_dict'])
optim.load_state_dict(checkpoint['optimizer'])

test_language_modeling(model, tokenizer)