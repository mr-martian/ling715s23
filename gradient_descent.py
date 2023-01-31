#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot

np.random.seed(3)

DIM = 2
SAMPLES = 100

input_data = np.random.rand(SAMPLES,DIM)
output_data = np.random.rand(SAMPLES)
weights = np.random.rand(DIM)

def update(W, X, Y, alpha):
    error = np.array([(X.dot(W)-Y).dot(X[:,i]) for i in range(W.shape[0])])
    norm = 1 / Y.shape[0]
    return W - (alpha * norm * error)

class Trainer:
    def __init__(self, W, X, Y, alpha, label):
        self.W = W
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.label = label
        self.cost_list = [self.cost()]
    def cost(self):
        m = (1/(2*self.Y.shape[0]))
        diff = self.X.dot(self.W) - self.Y
        return m * diff.dot(diff)
    def epoch(self):
        self.W = update(self.W, self.X, self.Y, self.alpha)
        self.cost_list.append(self.cost())
    def plot(self, ax):
        ax.plot(list(range(len(self.cost_list))), self.cost_list,
                label=self.label)

class MiniBatchTrainer(Trainer):
    def __init__(self, *args, batch_size=5):
        super().__init__(*args)
        self.batch_size = batch_size
    def minibatches(self):
        for i in range(0, self.Y.shape[0], self.batch_size):
            yield self.X[i:i+self.batch_size,:], self.Y[i:i+self.batch_size]
    def epoch(self):
        for x, y in self.minibatches():
            self.W = update(self.W, x, y, self.alpha)
        self.cost_list.append(self.cost())

class StochasticMiniBatchTrainer(MiniBatchTrainer):
    def minibatches(self):
        idx = list(range(self.Y.shape[0]))
        np.random.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            x = [self.X[j,:] for j in idx[i:i+self.batch_size]]
            y = [self.Y[j] for j in idx[i:i+self.batch_size]]
            yield np.array(x), np.array(y)

def do_epochs(trainers, EPOCHS):
    for i in range(1, EPOCHS+1):
        print(f'EPOCH {i}')
        for trainer in trainers:
            trainer.epoch()
    fig = pyplot.figure()
    ax = fig.add_subplot()
    for trainer in trainers:
        trainer.plot(ax)
    ax.legend()
    fig.savefig('gradient_descent.png')
            
trainers = [
    Trainer(weights[:], input_data, output_data, 0.1, "α = 0.1"),
    Trainer(weights[:], input_data, output_data, 0.01, "α = 0.01"),
    Trainer(weights[:], input_data, output_data, 0.001, "α = 0.001"),
    MiniBatchTrainer(weights[:], input_data, output_data, 0.01,
                     "minibatch 5, α = 0.01"),
    MiniBatchTrainer(weights[:], input_data, output_data, 0.01,
                     "minibatch 10, α = 0.01", batch_size=10),
    MiniBatchTrainer(weights[:], input_data, output_data, 0.01,
                     "minibatch 1, α = 0.01", batch_size=1),
    MiniBatchTrainer(weights[:], input_data, output_data, 0.1,
                     "minibatch 1, α = 0.1", batch_size=1),
    StochasticMiniBatchTrainer(weights[:], input_data, output_data, 0.01,
                               "stochastic minibatch 5, α = 0.01"),
]

do_epochs(trainers, 100)
