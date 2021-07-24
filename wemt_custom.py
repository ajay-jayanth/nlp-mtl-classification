import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer, TrainingArguments, Trainer
import pandas as pd
import emoji
import numpy as np
import sys
from custom_heads import full_conn, BiLSTM
from mtl_classifier import mtl_classifier, mtl_loss_wrapper
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
# from resample import resample
from train_args import args
from head_hyper_params import head_hyper_params
from fastai import *
import pdb
from fastai.learner import Learner

# Preparation
model_bert = AutoModel.from_pretrained('vinai/bertweet-covid19-base-cased')
tokenizer_bert = AutoTokenizer.from_pretrained('vinai/bertweet-covid19-base-cased')

file = 'concat-final-data.csv'
FILE_PATH = 'C:\\Users\\msctb\\Documents\\NLP Research\\' + file
trained_model_file = 'WEMT-CH-1'
dataset = pd.read_csv(FILE_PATH)
learn_set, test = train_test_split(dataset, test_size=0.2, shuffle=True)
train, val = train_test_split(learn_set, test_size=0.2, shuffle=True)
cats = [['first-hand', 'secondary'], ['general', 'local/personalized'], ['fact', 'opinion', 'question']]
# train = resample(train, cats)

# Preprocessing
def preprocess(data, tokenizer, max_length):
    tokenized_text = []
    len_list = []
    for i, text in enumerate(data['text']):
        encoding = tokenizer.encode(text)
        len_list.append(len(encoding))
        t = np.zeros(max_length-len(encoding))
        encoding.extend(list(t))
        tokenized_text.append(encoding)
    return tokenized_text

def label_encoding(data_series, cats, pool_sizes):
    labels = torch.tensor([])
    for cat, pool in zip(cats, pool_sizes):
        y = LabelEncoder().fit_transform(np.array(data_series[cat].values))
        label = torch.zeros((len(y), 1))
        label = torch.FloatTensor(y)
        labels = labels.cat((labels, label))
    return labels

class Data(Dataset):
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
        self.len = self.x.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

# Data
Encoded_matrix_train = torch.FloatTensor(preprocess(train, tokenizer_bert, 139))
Encoded_matrix_val = torch.FloatTensor(preprocess(val, tokenizer_bert, 139))
Encoded_matrix_test = torch.FloatTensor(preprocess(test, tokenizer_bert, 139))
tensor_y_train = label_encoding(train, ['content', 'audience', 'source'], [2, 2, 3])
tensor_y_val = label_encoding(val, ['content', 'audience', 'source'], [2, 2, 3])
tensor_y_test = label_encoding(test, ['content', 'audience', 'source'], [2, 2, 3])

train_dataset = Data(Encoded_matrix_train, tensor_y_train)
val_dataset = Data(Encoded_matrix_val, tensor_y_val)
test_dataset = Data(Encoded_matrix_test, tensor_y_test)
train_dl = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size)
val_dl = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size)
test_dl = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size)

# Metrics
def source_f1(pred, target): # Inputs of mtl_loss_wrapper
    cr = classification_report(target.T[0], pred.T[0], output_dict=True, zero_division=1)
    return cr['weighted avg']['f1-score']

def audience_f1(pred, target):
    cr = classification_report(target.T[1], pred.T[1], output_dict=True, zero_division=1)
    return cr['weighted avg']['f1-score']

def content_f1(pred, target):
    cr = classification_report(target.T[2], pred.T[2], output_dict=True, zero_division=1)
    return cr['weighted avg']['f1-score']

metrics = [source_f1, audience_f1, content_f1]

# Training and Optimization
model = mtl_classifier(model_bert, head_hyper_params=head_hyper_params)
loss_func = mtl_loss_wrapper(3)
learn = Learner((train_dl, val_dl), model, loss_func=loss_func, metrics=metrics)

learn.fit_one_cycle(args.num_train_epochs, max_lr=args.max_learning_rate)
learn.load('Stage 1/1')
trained_model = learn.model # learn.model.cpu() to bring back everything to cpu
torch.save(trained_model.state_dict(), trained_model_file)