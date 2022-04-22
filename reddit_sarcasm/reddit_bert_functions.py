import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import transformers
import json
from tqdm.notebook import tqdm
from transformers.utils.dummy_pt_objects import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification,AutoConfig, AutoModel,AutoTokenizer,BertModel,BertConfig,AdamW, get_constant_schedule,BertForSequenceClassification,get_linear_schedule_with_warmup
import random
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


class Reddit(Dataset):
    def __init__(self, pd_text, pd_labels, selected_tokenizer, max_length=None):
        
       
        self.inputs = selected_tokenizer.batch_encode_plus(pd_text.tolist(), max_length = max_length,\
                                                          padding = True, truncation = True, \
                                                         add_special_tokens = True, return_tensors = "pt", \
                                                          return_attention_mask = True)
        
        self.labels = torch.Tensor(pd_labels.tolist())
        return
        
    def __len__(self): 
        return len(self.labels)
        
        
    def __getitem__(self,item):
        text = {key: self.inputs[key][item] for key in self.inputs.keys()}
        label = self.labels[item]
        return text, label
    

def split_reddit_data(csv_path):
    
    #read in .csv
    data_all = None
    try:
        data_all = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        print('Data csv not found')
        return
    
    data_all.dropna(subset=['comment'], inplace=True)
    
    x_train, x_testval, y_train, y_testval= train_test_split(data_all['comment'], data_all['label'], random_state=200, 
                                                                    test_size=0.2, 
                                                                    stratify=data_all['label'])
    
    x_test, x_val, y_test, y_val = train_test_split(x_testval, y_testval, random_state=200, 
                                                                    test_size=0.5, 
                                                                    stratify=y_testval)
    
    return x_train, y_train, x_val, y_val, x_test, y_test 


def get_data_loaders(train, val, test, batch_size, num_workers):
    
    trainloader = DataLoader(train, batch_size = batch_size,num_workers=num_workers,shuffle = True)
    validationloader = DataLoader(val, batch_size = batch_size,num_workers=num_workers,shuffle = True)
    testloader = DataLoader(test, batch_size = batch_size,num_workers=num_workers,shuffle = True)
    
    return trainloader, validationloader, testloader


class bert_for_sarcasm(nn.Module):
    def __init__(self,input_model):
        super(bert_for_sarcasm,self).__init__()
        
        self.input_model = input_model
        self.fc1 = nn.Linear(self.input_model.config.hidden_size, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_values, attention_mask):
        hidden,out = self.input_model(input_values, attention_mask=attention_mask).values()
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.sigmoid(self.fc3(out))
        return out