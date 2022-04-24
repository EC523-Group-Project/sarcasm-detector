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


class bert_for_sarcasm(nn.Module):
    def __init__(self,input_model, dim_out = 1,linear1 = 256,linear2 = 128, drop = .25):
        super(bert_for_sarcasm,self).__init__()
        
        self.input_model = input_model
        self.fc1 = nn.Linear(self.input_model.config.hidden_size, linear1)
        self.fc2 = nn.Linear(linear1,linear2)
        self.fc3 = nn.Linear(linear2, dim_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_values, attention_mask):
        hidden,out = self.input_model(input_values, attention_mask=attention_mask).values()
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.sigmoid(self.fc3(out))
        return out