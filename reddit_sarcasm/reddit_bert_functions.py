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
import time
import datetime
import math


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



def validate_reddit(sarcasm_model, validationloader, loss_function, device):
    
    valid_loss_total = 0.0
    sarcasm_model.eval()
    val_correct = 0
    val_total = 0
    
    print("Validating.....")
    for encodings, labels in validationloader:
        inputs = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        if torch.cuda.is_available():
            labels = labels.to(device).float()
            inputs, attention_mask = inputs.to(device), attention_mask.to(device)
            
        preds = sarcasm_model(inputs, attention_mask)
        preds = torch.flatten(preds)
        
        loss = loss_function(preds,labels)
        valid_loss_total += loss.item()
        
        preds[preds<0.5] = 0
        preds[preds >=0.5] = 1
        
        val_correct += (preds == labels).float().sum().item()
        val_total += len(labels)
    
    valid_loss = valid_loss_total/len(validationloader)
    validation_acc = round(val_correct/val_total,4)
    
    return valid_loss, validation_acc


def train_reddit(sarcasm_model, trainloader, validationloader, epochs, batch_size, device, lr =1e-4, \
                 model_save_dir = "/projectnb/dl523/students/nannkat/Project/training/cp.ckpt" , \
                    scheduler = False, checkpoint = None):
    
    optimizer = torch.optim.AdamW(sarcasm_model.parameters(),lr = lr, eps = 1e-8)
    loss_function = nn.BCELoss()
    
    
    if scheduler:
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        lrs = []
    
    #for early stop
    last_val_loss = float('inf')
    best_val_loss = float('inf')
    
    patience = 3
    es_counter = 0
    
    if checkpoint != None:
        epoch_start = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss'].item()
    else:
        epoch_start = 1
        
    
    #prep
    losses = []
    val_losses = []
    start_time = time.time()
    num_iters = len(trainloader)
    
    #iterate over epoch
    for epoch in range(epoch_start, epochs+1):
    
        print('Epoch: ', epoch)
        train_iter = iter(trainloader)
        sarcasm_model.train()
        train_correct = 0
        train_total = 0

        if scheduler:
            print("Learning rate: ", scheduler.get_last_lr())
        
        #go through batches
        curr_loss = 0
        for idx, (encodings, labels) in enumerate(train_iter):

            labels = labels.to(device).float()

            inputs = encodings['input_ids']
            attention_mask = encodings['attention_mask']

            inputs, attention_mask = inputs.to(device), attention_mask.to(device)

            optimizer.zero_grad()

            output = sarcasm_model(inputs, attention_mask)
            output = torch.flatten(output)


            loss = loss_function(output,labels)
            curr_loss = loss.item()

            loss.backward()
            optimizer.step()
            #at first for hyp. tuning adjust scheduler every iteration
            if scheduler:
                lr_sched.step()
                lrs.append(optimizer.param_groups[0]["lr"])
                
            output[output<0.5] = 0
            output[output>=0.5] = 1
            train_correct += (output == labels).float().sum().item()
            train_total += len(labels)

            if idx%2000 == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, idx+1, num_iters)
                log += "Loss: {:.4f}".format(loss)
                print(log)
        
        #AT THE END OF EACH EPOCH
        
        #print("Current learning rate: {}".format(lrs[-1]))
        
        #validate and get accuracies
        losses.append(curr_loss)
        
        training_acc = round(train_correct/train_total,4)
        curr_val_loss, validation_acc = validate_reddit(sarcasm_model, validationloader, loss_function, device)
        val_losses.append(curr_val_loss)
        
        #early stopping
        if curr_val_loss >= last_val_loss:
                es_counter += 1

                if es_counter >= patience:
                    print("Early stopping triggered. Ending training..")
                    return
                else:
                    print(f"No decrease in validation loss! {patience-es_counter} more consecutive loss increase(s) until early stop.")

        else:
            print("Decrease in validation loss. Early stop counter reset to 0.")
            es_counter = 0
            
            last_val_loss = curr_val_loss
            
            if curr_val_loss < best_val_loss:
                best_val_loss = curr_val_loss
            
                print("New lowest loss, saving model...")
                torch.save({'epoch': epoch,
                    'model_state_dict': sarcasm_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, model_save_dir)
    
                print("Model checkpoint saved to " + model_save_dir)
                
            
        
        print("Epoch {}. Training accuracy: {}. Validation accuracy: {}.".format(epoch, training_acc, validation_acc))
        print()
    
    return losses, val_losses


def freeze_by_children(model, number_to_unfreeze):
    # Freezes model elements
    encoding_layers = list(list(model.children())[1].children())[0]
    pooling_layer = list(model.children())[2]
    print('The input model has',len(encoding_layers),'encoding layers')
    print('The model has 1 pooling layers')
    
    n = len(encoding_layers)
    if number_to_unfreeze <= n:
        for i, child in enumerate(encoding_layers):
            if i < n-number_to_unfreeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
                print('Bert layer {} has been unfrozen'.format(i+1))
    else:
        print('Invalid Number')

    for param in pooling_layer.parameters():
        param.requires_grad = True
    print("Pooling layer has been unfrozen")