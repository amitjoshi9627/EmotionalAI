import numpy as np
import os
import pickle
import transformers


def train_test_split(X, y, test_size=0.3, train_size=None, random_state=None):

    if random_state is not None:
        np.random.seed(seed=random_state)
    if train_size:
        test_size = 1.0 - train_size
    X = np.array(X)
    y = np.array(y)
    size = X.shape[0]
    indx = np.random.choice(size, int(size * test_size))

    return X[~indx], X[indx], y[~indx], y[indx]


def accuracy_score(data1, data2):
    return np.mean(np.array(data1) == np.array(data2))


def save_model(model, model_path):
    if not os.path.exists("Saved_Model/"):
        os.makedirs('Saved_Model/')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


def load_model(pkl_filename):
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model

def load_bert_model(bert_model_path):
    if not os.path.exists(bert_model_path):
        print("Downloading Model...")
        tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        os.makedirs(bert_model_path)
        tokenizer.save_pretrained(bert_model_path)
        model.save_pretrained(bert_model_path)
    else:
        print("Loading Model...")
        model = transformers.DistilBertModel.from_pretrained(bert_model_path)
    return model

def load_bert_tokenizer(tokenizer_path):
    if not os.path.exists(tokenizer_path):
        tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        tokenizer = transformers.DistilBertTokenizer.from_pretrained(tokenizer_path)
    return tokenizer