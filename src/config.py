import transformers
import os

TRAIN_URL = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv'

DIR_ROOT = os.getcwd()
TRAIN_DATA_PATH = os.path.join(DIR_ROOT, "../data/train.csv")

BERT_PATH = os.path.join(DIR_ROOT, "../Distilbert_base_uncased")

BERT_DOWNLOAD_PATH = 'distilbert-base-uncased'

TOKENIZER = transformers.DistilBertTokenizer.from_pretrained(
    BERT_DOWNLOAD_PATH)
MODEL_PATH = os.path.join(DIR_ROOT, "../Saved_Model/svm_model.pkl")

MAX_LEN = 128
BATCH_SIZE = 2
