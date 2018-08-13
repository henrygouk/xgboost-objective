import argparse
import ml_dataset_loader.datasets as data_loader
import numpy as np
import xgboost as xgb

parser = argparse.ArgumentParser(description="Run experiment for comparing hinge loss and logistic loss")
parser.add_argument("--dataset")
parser.add_argument("--objective")
parser.add_argument("--eta")
args = parser.parse_args()

num_rounds = 5000
learning_rate = args.eta
dataset = args.dataset
objective = "binary:" + args.objective

params = {"silent": 1, "tree_method": "hist", "split_evaluator": "elastic_net", "learning_rate": learning_rate, "objective": objective}


class Experiment:
    def __init__(self, loader, num_train):
        self.loader = loader
        self.num_train = num_train
    
    def run(self):
        X, y = self.loader()
        X_train = X[:self.num_train]
        y_train = y[:self.num_train]
        X_test = X[self.num_train:]
        y_test = y[self.num_train:]

        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)
        bst = xgb.train(params, dtrain, num_rounds, evals=[(dtrain, "train"), (dtest, "test")])


if dataset == "higgs":
    exp = Experiment(data_loader.get_higgs, 11000000 - 500000)
elif "synthetic":
    exp = Experiment(data_loader.get_synthetic_classification, 9000000)
else:
    raise Exception("Unknown dataset: " + dataset)

exp.run()
