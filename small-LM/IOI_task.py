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

import scipy

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

classic_model_files = ['models/model_8M_1003_185803.pt.tar', 'models/model_8M_1005_115351.pt.tar', 'models/model_8M_1006_171728.pt.tar']
modified_model_files = ['models/model_8M_1030_175253.pt.tar', 'models/model_8M_1101_120834.pt.tar', 'models/model_8M_1102_182840.pt.tar']

for condition in ["classic","modified"]:
    print("condition:",condition)
    if condition == "classic":
        model_files = classic_model_files
    elif condition == "modified":
        model_files = modified_model_files
    
    sentence1_correct_probs = []
    sentence2_correct_probs = []
    sentence3_correct_probs = []
    sentence4_correct_probs = []
    
    sentence1_incorrect_probs = []
    sentence2_incorrect_probs = []
    sentence3_incorrect_probs = []
    sentence4_incorrect_probs = []
    
    for model_file in model_files:
        print("model file:",model_file)

        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        
        # sentence 1
        
        prompt1 = "After John and Mary went to the store, John gave a bottle of milk to"
        input_ids1 = tokenizer.encode(prompt1, return_tensors="pt").to(device)
        greedy_output1, logits1, probs1 = model.module.generate(input_ids1, pad_token_id=tokenizer.pad_token_id, max_length=1)
        text_output1 = tokenizer.decode(greedy_output1[0][-1:], skip_special_tokens=True)
        correct_ID1 = tokenizer.encode(" Mary", return_tensors="pt").to(device)
        incorrect_ID1 = tokenizer.encode(" John", return_tensors="pt").to(device)
        
        prompt2 = "After John and Mary went to the store, Mary gave a bottle of milk to"
        input_ids2 = tokenizer.encode(prompt2, return_tensors="pt").to(device)
        greedy_output2, logits2, probs2 = model.module.generate(input_ids2, pad_token_id=tokenizer.pad_token_id, max_length=1)
        text_output2 = tokenizer.decode(greedy_output2[0][-1:], skip_special_tokens=True)
        correct_ID2 = tokenizer.encode(" John", return_tensors="pt").to(device)
        incorrect_ID2 = tokenizer.encode(" Mary", return_tensors="pt").to(device)
        
        correct_logits12 = (logits1[0,correct_ID1[0]] + logits2[0,correct_ID2[0]]) / 2
        correct_probs12 = (probs1[0,correct_ID1[0]] + probs2[0,correct_ID2[0]]) / 2
        
        incorrect_logits12 = (logits1[0,incorrect_ID1[0]] + logits2[0,incorrect_ID2[0]]) / 2
        incorrect_probs12 = (probs1[0,incorrect_ID1[0]] + probs2[0,incorrect_ID2[0]]) / 2
        
        sentence1_correct_probs.append(correct_probs12)
        sentence1_incorrect_probs.append(incorrect_probs12)
        
        # print("sentence 1")
        # print("correct logits + probs", correct_logits12, correct_probs12)
        # print("incorrect logit + prob", incorrect_logits12, incorrect_probs12)
        
        # sentence 2
        
        prompt1 = "When Tom and James went to the park, Tom gave the ball to"
        input_ids1 = tokenizer.encode(prompt1, return_tensors="pt").to(device)
        greedy_output1, logits1, probs1 = model.module.generate(input_ids1, pad_token_id=tokenizer.pad_token_id, max_length=1)
        text_output1 = tokenizer.decode(greedy_output1[0][-1:], skip_special_tokens=True)
        correct_ID1 = tokenizer.encode(" James", return_tensors="pt").to(device)
        incorrect_ID1 = tokenizer.encode(" Tom", return_tensors="pt").to(device)
        
        prompt2 = "When Tom and James went to the park, James gave the ball to"
        input_ids2 = tokenizer.encode(prompt2, return_tensors="pt").to(device)
        greedy_output2, logits2, probs2 = model.module.generate(input_ids2, pad_token_id=tokenizer.pad_token_id, max_length=1)
        text_output2 = tokenizer.decode(greedy_output2[0][-1:], skip_special_tokens=True)
        correct_ID2 = tokenizer.encode(" Tom", return_tensors="pt").to(device)
        incorrect_ID2 = tokenizer.encode(" James", return_tensors="pt").to(device)
        
        correct_logits12 = (logits1[0,correct_ID1[0]] + logits2[0,correct_ID2[0]]) / 2
        correct_probs12 = (probs1[0,correct_ID1[0]] + probs2[0,correct_ID2[0]]) / 2
        
        incorrect_logits12 = (logits1[0,incorrect_ID1[0]] + logits2[0,incorrect_ID2[0]]) / 2
        incorrect_probs12 = (probs1[0,incorrect_ID1[0]] + probs2[0,incorrect_ID2[0]]) / 2
        
        sentence2_correct_probs.append(correct_probs12)
        sentence2_incorrect_probs.append(incorrect_probs12)
        
        # print("sentence 2")
        # print("correct logits + probs", correct_logits12, correct_probs12)
        # print("incorrect logit + prob", incorrect_logits12, incorrect_probs12)
        
        # sentence 3
        
        prompt1 = "When Dan and Emily went to the shops, Dan gave an apple to"
        input_ids1 = tokenizer.encode(prompt1, return_tensors="pt").to(device)
        greedy_output1, logits1, probs1 = model.module.generate(input_ids1, pad_token_id=tokenizer.pad_token_id, max_length=1)
        text_output1 = tokenizer.decode(greedy_output1[0][-1:], skip_special_tokens=True)
        correct_ID1 = tokenizer.encode(" Emily", return_tensors="pt").to(device)
        incorrect_ID1 = tokenizer.encode(" Dan", return_tensors="pt").to(device)
        
        prompt2 = "When Dan and Emily went to the shops, Emily gave an apple to"
        input_ids2 = tokenizer.encode(prompt2, return_tensors="pt").to(device)
        greedy_output2, logits2, probs2 = model.module.generate(input_ids2, pad_token_id=tokenizer.pad_token_id, max_length=1)
        text_output2 = tokenizer.decode(greedy_output2[0][-1:], skip_special_tokens=True)
        correct_ID2 = tokenizer.encode(" Dan", return_tensors="pt").to(device)
        incorrect_ID2 = tokenizer.encode(" Emily", return_tensors="pt").to(device)
        
        correct_logits12 = (logits1[0,correct_ID1[0]] + logits2[0,correct_ID2[0]]) / 2
        correct_probs12 = (probs1[0,correct_ID1[0]] + probs2[0,correct_ID2[0]]) / 2
        
        incorrect_logits12 = (logits1[0,incorrect_ID1[0]] + logits2[0,incorrect_ID2[0]]) / 2
        incorrect_probs12 = (probs1[0,incorrect_ID1[0]] + probs2[0,incorrect_ID2[0]]) / 2
        
        sentence3_correct_probs.append(correct_probs12)
        sentence3_incorrect_probs.append(incorrect_probs12)
        
        # print("sentence 3")
        # print("correct logits + probs", correct_logits12, correct_probs12)
        # print("incorrect logit + prob", incorrect_logits12, incorrect_probs12)
        
        # sentence 4
        
        prompt1 = "After Sam and Amy went to the park, Sam gave a drink to"
        input_ids1 = tokenizer.encode(prompt1, return_tensors="pt").to(device)
        greedy_output1, logits1, probs1 = model.module.generate(input_ids1, pad_token_id=tokenizer.pad_token_id, max_length=1)
        text_output1 = tokenizer.decode(greedy_output1[0][-1:], skip_special_tokens=True)
        correct_ID1 = tokenizer.encode(" Amy", return_tensors="pt").to(device)
        incorrect_ID1 = tokenizer.encode(" Sam", return_tensors="pt").to(device)
        
        prompt2 = "After Sam and Amy went to the park, Amy gave a drink to"
        input_ids2 = tokenizer.encode(prompt2, return_tensors="pt").to(device)
        greedy_output2, logits2, probs2 = model.module.generate(input_ids2, pad_token_id=tokenizer.pad_token_id, max_length=1)
        text_output2 = tokenizer.decode(greedy_output2[0][-1:], skip_special_tokens=True)
        correct_ID2 = tokenizer.encode(" Sam", return_tensors="pt").to(device)
        incorrect_ID2 = tokenizer.encode(" Amy", return_tensors="pt").to(device)
        
        correct_logits12 = (logits1[0,correct_ID1[0]] + logits2[0,correct_ID2[0]]) / 2
        correct_probs12 = (probs1[0,correct_ID1[0]] + probs2[0,correct_ID2[0]]) / 2
        
        incorrect_logits12 = (logits1[0,incorrect_ID1[0]] + logits2[0,incorrect_ID2[0]]) / 2
        incorrect_probs12 = (probs1[0,incorrect_ID1[0]] + probs2[0,incorrect_ID2[0]]) / 2
        
        sentence4_correct_probs.append(correct_probs12)
        sentence4_incorrect_probs.append(incorrect_probs12)
        
        # print("sentence 4")
        # print("correct logits + probs", correct_logits12, correct_probs12)
        # print("incorrect logit + prob", incorrect_logits12, incorrect_probs12)
    
    print("overall condition probs")
    print("sentence 1")
    print("mean + std correct:",torch.mean(torch.tensor(sentence1_correct_probs)),torch.std(torch.tensor(sentence1_correct_probs)))
    print("mean + std incorrect:",torch.mean(torch.tensor(sentence1_incorrect_probs)),torch.std(torch.tensor(sentence1_incorrect_probs)))
    print("sentence 2")
    print("mean + std correct:",torch.mean(torch.tensor(sentence2_correct_probs)),torch.std(torch.tensor(sentence2_correct_probs)))
    print("mean + std incorrect:",torch.mean(torch.tensor(sentence2_incorrect_probs)),torch.std(torch.tensor(sentence2_incorrect_probs)))
    print("sentence 3")
    print("mean + std correct:",torch.mean(torch.tensor(sentence3_correct_probs)),torch.std(torch.tensor(sentence3_correct_probs)))
    print("mean + std incorrect:",torch.mean(torch.tensor(sentence3_incorrect_probs)),torch.std(torch.tensor(sentence3_incorrect_probs)))
    print("sentence 4")
    print("mean + std correct:",torch.mean(torch.tensor(sentence4_correct_probs)),torch.std(torch.tensor(sentence4_correct_probs)))
    print("mean + std incorrect:",torch.mean(torch.tensor(sentence4_incorrect_probs)),torch.std(torch.tensor(sentence4_incorrect_probs)))
    
    if condition == "classic":
        classic_results = [torch.tensor(sentence1_correct_probs).numpy(), torch.tensor(sentence1_incorrect_probs).numpy(), 
                           torch.tensor(sentence2_correct_probs).numpy(), torch.tensor(sentence2_incorrect_probs).numpy(), 
                           torch.tensor(sentence3_correct_probs).numpy(), torch.tensor(sentence3_incorrect_probs).numpy(), 
                           torch.tensor(sentence4_correct_probs).numpy(), torch.tensor(sentence4_incorrect_probs).numpy()]
    elif condition == "modified":
        modified_results = [torch.tensor(sentence1_correct_probs).numpy(), torch.tensor(sentence1_incorrect_probs).numpy(), 
                           torch.tensor(sentence2_correct_probs).numpy(), torch.tensor(sentence2_incorrect_probs).numpy(), 
                           torch.tensor(sentence3_correct_probs).numpy(), torch.tensor(sentence3_incorrect_probs).numpy(), 
                           torch.tensor(sentence4_correct_probs).numpy(), torch.tensor(sentence4_incorrect_probs).numpy()]

print("t tests")
print("sentence 1:", scipy.stats.ttest_rel(classic_results[0],modified_results[0]))
print("sentence 2:", scipy.stats.ttest_rel(classic_results[1],modified_results[1]))
print("sentence 3:", scipy.stats.ttest_rel(classic_results[2],modified_results[2]))
print("sentence 4:", scipy.stats.ttest_rel(classic_results[3],modified_results[3]))

# empirical verficiation
#
# correct_name1 = 0
# wrong_name1 = 0
# for _ in range(1000):
#     prompt = "After John and Mary went to the store, John gave a bottle of milk to"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     greedy_output = model.module.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_length=1)
#     text_output = tokenizer.decode(greedy_output[0][-1:], skip_special_tokens=True)
#     if text_output == " Mary":
#         correct_name1 += 1
#     elif text_output == " John":
#         wrong_name1 += 1
        
# correct_name2 = 0
# wrong_name2 = 0
# for _ in range(1000):
#     prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     greedy_output = model.module.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_length=1)
#     text_output = tokenizer.decode(greedy_output[0][-1:], skip_special_tokens=True)
#     if text_output == " John":
#         correct_name2 += 1
#     elif text_output == " Mary":
#         wrong_name2 += 1

# correct_name3 = 0
# wrong_name3 = 0
# for _ in range(1000):
#     prompt = "When Tom and James went to the park, Tom gave the ball to"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     greedy_output = model.module.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_length=1)
#     text_output = tokenizer.decode(greedy_output[0][-1:], skip_special_tokens=True)
#     if text_output == " James":
#         correct_name3 += 1
#     elif text_output == " Tom":
#         wrong_name3 += 1
    
# correct_name4 = 0
# wrong_name4 = 0
# for _ in range(1000):
#     prompt = "When Tom and James went to the park, James gave the ball to"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     greedy_output = model.module.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_length=1)
#     text_output = tokenizer.decode(greedy_output[0][-1:], skip_special_tokens=True)
#     if text_output == " Tom":
#         correct_name4 += 1
#     elif text_output == " James":
#         wrong_name4 += 1

# correct_name5 = 0
# wrong_name5 = 0
# for _ in range(1000):
#     prompt = "When Dan and Emily went to the shops, Dan gave an apple to"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     greedy_output = model.module.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_length=1)
#     text_output = tokenizer.decode(greedy_output[0][-1:], skip_special_tokens=True)
#     if text_output == " Emily":
#         correct_name5 += 1
#     elif text_output == " Dan":
#         wrong_name5 += 1

# correct_name6 = 0
# wrong_name6 = 0
# for _ in range(1000):
#     prompt = "When Dan and Emily went to the shops, Emily gave an apple to"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     greedy_output = model.module.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_length=1)
#     text_output = tokenizer.decode(greedy_output[0][-1:], skip_special_tokens=True)
#     if text_output == " Dan":
#         correct_name6 += 1
#     elif text_output == " Emily":
#         wrong_name6 += 1
    
# correct_name7 = 0
# wrong_name7 = 0
# for _ in range(1000):
#     prompt = "After Sam and Amy went to the park, Sam gave a drink to"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     greedy_output = model.module.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_length=1)
#     text_output = tokenizer.decode(greedy_output[0][-1:], skip_special_tokens=True)
#     if text_output == " Amy":
#         correct_name7 += 1
#     elif text_output == " Sam":
#         wrong_name7 += 1

# correct_name8 = 0
# wrong_name8 = 0
# for _ in range(1000):
#     prompt = "After Sam and Amy went to the park, Amy gave a drink to"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     greedy_output = model.module.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_length=1)
#     text_output = tokenizer.decode(greedy_output[0][-1:], skip_special_tokens=True)
#     if text_output == " Sam":
#         correct_name8 += 1
#     elif text_output == " Amy":
#         wrong_name8 += 1

# print(correct_name1)
# print(wrong_name1)
# print(correct_name2)
# print(wrong_name2)
# print(correct_name3)
# print(wrong_name3)
# print(correct_name4)
# print(wrong_name4)
# print(correct_name5)
# print(wrong_name5)
# print(correct_name6)
# print(wrong_name6)
# print(correct_name7)
# print(wrong_name7)
# print(correct_name8)
# print(wrong_name8)