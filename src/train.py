import config
from model import BertForFeatureExtraction
import engine
import utils
import pandas as pd


def _training():
    Model = BertForFeatureExtraction()
    features, targets = engine.get_features(Model, train=True)
    X_train, X_test, y_train, y_test = utils.train_test_split(
        features, targets, test_size=0.3)
    classifier = engine.train_fn(X_train, y_train)
    predictions = engine.eval_fn(classifier, X_test)
    accuracy = utils.accuracy_score(predictions, y_test)
    print("Accuracy Score:", accuracy)
    classifier.fit(X_test, y_test)
    utils.save_model(classifier, config.MODEL_PATH)


def run():
    _training()


if __name__ == "__main__":
    run()
