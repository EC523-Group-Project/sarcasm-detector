# sarcasm-detector
Implementation of a sarcasm detection model using deep learning with BERT to detect sarcasm in news headlines and social media posts

By: Nanna Hannesdottir, Kevin Vogt-Lowell, Cole Hunter

# File Setup and How to Run

### Installations and Libraries neededed
  - PyTorch 1.11
  - Cuda 11.2
  - numpy, pandas, matplotlib
  - io,os
  - sklearn
  - optuna
  - tqdm
  - json
 
Additionally, we used multiple methods from the Huggingface transformers library (installations included in notebooks).

## Datasets

We used the following two datasets in our experimentation, both available on Kaggle.

1. SARC Reddit data: https://www.kaggle.com/datasets/danofer/sarcasm**
2. News Headlines data: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

The News Headlines dataset can also be found in this repository under the data/ folder. The Reddit data is too big to upload to Github directly.

Under data/ you can also find a notebook with our preliminary inspections of the datasets. This notebook is exploratory only and not required to replicate the results.

## reddit_sarcasm/ folder: Reddit models and training

### BERT with basic linear head
  - Main script for training and testing: basic_BERT_reddit.ipynb 
  - Functions and classes: defined in reddit_bert_functions.py, bert_sarcasm_model.py

### BERT with multihead self-attention head
  - Main script with all functionality: multihead_bert_reddit.ipynb. Just run cells in order.

## headlines_sarcasm/folder: Headlines models and training

### BERT with basic linear head
   - Main script including all classes and functions for training and testing: Headlines_model.ipynb
### BERT with multihead self-attention head
   - Main script with all functionality: multihead_bert_headlines.ipynb. Just run cells in order.

**NOTE: For reddit, we exclusively used train-balanced-sarcasm.csv as our data. The Reddit dataset is split into training and testing sets, so we had planned on using the data splits as provided. However, a more in-depth analysis of the dataset and its splits revealed that the data contained within the test set seemed to bear no relation in context or format to the data present in the training set. To resolve the issue, we decided that, given the massive size of the dataset, we would simply treat the training set as our main Reddit dataset and then create our training, testing, and validation splits from this newly defined set. Judging by notebooks available on Kaggle, other researchers who used this dataset frequently replicated this approach.
