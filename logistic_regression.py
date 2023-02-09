#!/usr/bin/env python3

import numpy as np
from sklearn import datasets
from matplotlib import pyplot

class Model:
    def __init__(self, W, B):
        self.W = W
        self.B = B
    def sigma(self, vec):
        return np.reciprocal(np.exp(-vec)+1)
    def loss(self, y_hat, Y):
        return -np.sum(np.multiply(np.log(y_hat), Y)) / len(Y)
    def predict(self, X):
        return self.sigma(np.matmul(X, np.transpose(self.W)) + self.B)
    def update(self, X, Y, alpha):
        y_hat = self.predict(X)
        cur_loss = self.loss(y_hat, Y)
        m = alpha / len(X)
        diff = y_hat - Y
        self.B -= m * np.sum(diff, axis=0)
        self.W -= m * np.matmul(np.transpose(diff), X)
        return cur_loss

class Trainer:
    def __init__(self, model, X, Y, alpha, label, batch_size=-1):
        self.model = model
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.label = label
        self.batch_size = batch_size
        self.costs = []
    def batches(self):
        for i in range(0, self.Y.shape[0], self.batch_size):
            yield self.X[i:i+self.batch_size,:], self.Y[i:i+self.batch_size]
    def accuracy(self):
        y_hat = self.model.predict(self.X)
        good = 0
        for yi, yhi in zip(self.Y, y_hat):
            if np.argmax(yi) == np.argmax(yhi):
                good += 1
        return good / len(y_hat)
    def epoch(self):
        if self.batch_size > 0:
            self.costs.append(self.model.get_cost(self.X, self.Y))
            for x, y in self.batches():
                self.model.update(x, y, self.alpha)
        else:
            self.costs.append(self.model.update(self.X, self.Y, self.alpha))
    def plot(self, ax):
        ax.plot(list(range(len(self.costs))), self.costs, label=self.label)

def do_epochs(trainers, EPOCHS):
    for i in range(1, EPOCHS+1):
        print(f'EPOCH {i}')
        for trainer in trainers:
            trainer.epoch()
    fig = pyplot.figure()
    ax = fig.add_subplot()
    ax.clear()
    for trainer in trainers:
        trainer.plot(ax)
    ax.legend()
    fig.savefig('logistic_regression.png')

FEATS = 3
CLASSES = 5
EXAMPLES = 100
    
X, y_0 = datasets.make_blobs(n_samples=EXAMPLES, n_features=FEATS, centers=CLASSES, cluster_std=1.05, random_state=3)

Y = np.zeros((EXAMPLES, CLASSES))
for i, y in enumerate(y_0):
    Y[i,y] = 1

W = np.random.rand(CLASSES, FEATS)
B = np.random.rand(CLASSES)

trainers = [
    Trainer(Model(W[:,:], B[:]), X, Y, 0.1, "α = 0.1"),
    Trainer(Model(W[:,:], B[:]), X, Y, 0.01, "α = 0.01"),
    Trainer(Model(W[:,:], B[:]), X, Y, 0.001, "α = 0.001"),
]

do_epochs(trainers, 5)

# sanity check against page 17 of the textbook
#x = np.array([[3, 2]])
#y = np.array([[1]])
#m = Model(np.zeros((1, 2)), np.zeros((1,)))
#m.update(x, y, 0.1)
#print(m.W) # [[0.15 0.1 ]]
#print(m.B) # [0.05]
