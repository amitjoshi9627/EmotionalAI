import torch
import numpy as np
from model import BertForFeatureExtraction
from distilbert import BertForSentimentAnalysis
import utils
import os
import engine
import config


def get_prediction(text):
    Bert_model = SentimentClassifier()
    prediction = Bert_model.predict(text)
    print("\nPrediction:", prediction, "\n")
    if prediction:
        return True
    else:
        return False


class SentimentClassifier():
    def __init__(self):
        self.model = BertForFeatureExtraction()

    def predict(self, text):
        clf = utils.load_model(config.MODEL_PATH)
        test_features = engine.get_features(self.model, train=False, text=text)
        test_predictions = engine.test_results(clf, test_features)
        return test_predictions[0]
