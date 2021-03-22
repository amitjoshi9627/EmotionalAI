from tqdm import tqdm
from dataset import SentimentClassifierDataset
from torch.utils.data import DataLoader
import config
import torch
import numpy as np
from sklearn import svm


def get_features(model, train=True, text=None):
    if train == True:
        dataset = SentimentClassifierDataset()
        n_samples = len(dataset)
        train_dataloader = DataLoader(
            dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        labels = []
        features = []
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        model.to(device)
        with torch.no_grad():
            for ind, d in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                input_ids = d['input_ids'].to(device).long()
                attention_mask = d['attention_mask'].to(device).long()
                label = np.array(d['label'])
                labels.extend(label)
                feature = model(input_ids, attention_mask)
                features.extend(feature)
                del ind
                del d
        return np.array(features), np.array(labels)

    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        model.to(device)
        dataset = SentimentClassifierDataset(train=False, text=text)
        input_ids = [dataset.get_tokens(dataset.data)]
        attention_mask = dataset.get_attention_mask(np.array(input_ids[0]))

        with torch.no_grad():
            input_ids = torch.tensor(input_ids).to(device).long()
            attention_mask = torch.tensor(attention_mask).to(device).long()
            feature = model(input_ids, attention_mask)

        return np.array(feature)


def train_fn(X_train, y_train):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    return clf


def eval_fn(clf, X_test):
    return clf.predict(X_test)


def test_results(clf, test_data):
    return clf.predict(test_data)
