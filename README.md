# sarcasm-detector
Implementation of a sarcasm detection model using deep learning with BERT to detect sarcasm in news headlines and social media posts

By: Nanna Hannesdottir, Kevin Vogt-Lowell, Cole Hunter

# File Setup and How to Run
The contents of the repository can be roughly divided into 3 sections: 
1. Data 
2. Our models and experimentation for the Reddit dataset and 
3. Our models and experimentation for the Headlines dataset.

### Installations and Libraries neededed
  - PyTorch 1.11
  - Cuda 11.2
  - numpy, pandas, matplotlib
  - io,os
  - sklearn
  - optuna
  - tqdm
  - json
  - .......
 
Additionally we used multiple methods from the Huggingface transformers library, installations included in notebooks.

## 1. Datasets

We used the following two datasets in our experimentation, both available on Kaggle.

1. SARC Reddit data: https://www.kaggle.com/datasets/danofer/sarcasm**
2. News Headlines data: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

The News Headlines dataset can also be found on this under the data/ folder. The Reddit data is too big to upload to Github directly.

Under data/ you can furthermore find a notebook with our preliminary inspections of the datasets. This notebook is exploratory only and not required to replicate the results.

## 2. reddit_sarcasm/ folder: Reddit models and training

### BERT with basic linear head
  - Main script for training and testing: basic_BERT_reddit.ipynb 
  - Functions and classes: defined in reddit_bert_functions.py, bert_sarcasm_model.py

### BERT with multi-attention head
  -TODO: insert!!!!

## 3. headlines_sarcasm/folder: Headlines models and training

### BERT with basic linear head
   -Main script including all classes and functions for training and testing: Headlines_model.ipynb
### BERT with multi-attention head
   -TODO: insert!!!!

**NOTE: For reddit, we exclusively used train-balanced-sarcasm.csv as our data. The Reddit dataset is split into training and testing sets, so we had planned on using the data splits as provided. However, a more in-depth analysis of the dataset and its splits revealed that the data contained within the test set seemed to bear no relation in context or format to the data present in the training set. To resolve the issue, we decided that, given the massive size of the dataset, we would simply treat the training set as our main Reddit dataset and then create our training, testing, and validation splits from this newly defined set. Judging by notebooks available on Kaggle, other researchers who used this dataset frequently replicated this approach.
