import torch
import pandas as pd
from os import walk
import numpy as np
import json
import pickle
import bz2
import _pickle as cPickle
import importlib
import sys
import transformers as ppb
from transformers import pipeline
import re
from sklearn.metrics import confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

def compare_sent_transformer(model,sent1,sent2):
    sentences = [sent1,sent2]
    embeddings = model.encode(sentences)
    results = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )
    results = results.flatten()
    return results[0]

def compare_bert_average(model,tokenizer,sent1,sent2):
    # https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
    max_length = 0
    ls_sent = [sent1,sent2]
    for s in ls_sent:
        if len(s) > max_length:
            max_length = len(s)
   # initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}
    for sentence in ls_sent:
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=max_length,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    # mean pooling
    # ignore 0 attention mask
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    # calculate the mean as the sum of the embedding activations summed divided by the number of values that should be given attention in each position summed_mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled = mean_pooled.detach().numpy()
    # calculate
    sim_score = cosine_similarity(
        [mean_pooled[0]],
        [mean_pooled[1]])
    sim_score = sim_score.flatten()[0]
    return sim_score

def compare_bert_cls(model,tokenizer,sent1,sent2):
    # https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
    max_length = 0
    ls_sent = [sent1,sent2]
    for s in ls_sent:
        if len(s) > max_length:
            max_length = len(s)
    # initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}
    for sentence in ls_sent:
        # encode each sentence and append to dictionary
        # no need to specify add_special_tokens=True, but added for consistency, xiaoou
        new_tokens = tokenizer.encode_plus(sentence, max_length=max_length,add_special_tokens=True,
                                            truncation=True, padding='max_length',
                                            return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    first_sent_cls = embeddings[0][0][:]
    second_sent_cls = embeddings[1][0][:]
    return cosine_similarity(
        [first_sent_cls.detach().numpy()],
        [second_sent_cls.detach().numpy()]
    ).flatten().flatten()[0]