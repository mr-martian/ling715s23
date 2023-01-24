#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot, lines, animation

class BinaryPerceptron:
    def __init__(self, dim=2):
        self.dim = dim
        self.weights = np.random.rand(dim)
        self.bias = np.random.random()
    def eval(self, instance):
        return 1 if self.weights.dot(instance) + self.bias > 0 else 0
    def update(self, instance, expected, alpha=0.01):
        cur = self.eval(instance)
        if cur != expected:
            sign = 1 if cur == 0 else -1
            self.weights += instance * alpha * sign
            self.bias += alpha * sign

class DataSet:
    def __init__(self, examples, dim):
        self.examples = examples # (np.array, val)
        self.dim = dim
    def from_tsv(fname):
        with open(fname) as fin:
            examples = []
            for line in fin:
                if not line.strip() or 'output' in line:
                    continue
                ls = line.strip().split('\t')
                arr = np.array([float(c.strip()) for c in ls[:-1]])
                out = int(ls[-1])
                examples.append((arr, out))
            if not examples:
                raise ValueError('No input examples found')
            if len(set(x[0].shape for x in examples)) != 1:
                raise ValueError('Input examples have varying numbers of features')
            return DataSet(examples, examples[0][0].shape[0])

def epoch(percep, data, alpha=0.01):
    for inst, out in data.examples:
        percep.update(inst, out, alpha)

def print_accuracy(percep, data):
    correct = sum(1 if percep.eval(i) == o else 0 for i,o in data.examples)
    total = len(data.examples)
    rate = round(100*correct/total, 2)
    print(f'{correct} / {total} = ~{rate}% correct')
    return rate

class DataPlot:
    def __init__(self, data):
        x, y, c = list(zip(*[[i[0], i[1], o] for i, o in data.examples]))
        self.xdata = x
        self.ydata = y
        self.cdata = c
        self.maxx = max(x)
        self.minx = min(x)
        self.maxy = max(y)
        self.miny = min(y)
        self.y = []
        self.accuracy = []
    def record_epoch(self, percep, rate):
        y0 = (-percep.bias - self.minx*percep.weights[0]) / percep.weights[1]
        y1 = (-percep.bias - self.maxx*percep.weights[1]) / percep.weights[1]
        self.y.append([y0, y1])
        self.accuracy.append(rate)
    def animate(self, outfname):
        fig = pyplot.figure()
        ax = fig.add_subplot()
        def acb(i):
            ax.clear()
            ax.set_xlim([self.minx-0.5, self.maxx+0.5])
            ax.set_ylim([self.miny-0.5, self.maxy+0.5])
            ax.scatter(self.xdata, self.ydata, c=self.cdata)
            ax.plot([self.minx, self.maxx], self.y[i])
            ax.set_title(f'epoch {i} {self.accuracy[i]}%')
        anim = animation.FuncAnimation(fig, acb, frames=len(self.y))
        anim.save(outfname)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('train a moderately dumb perceptron')
    parser.add_argument('datafile', action='store')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-a', '--alpha', type=float, default=0.01,
                        help='learning rate (default 0.01)')
    parser.add_argument('-v', '--video', action='store', default=None,
                        help='output file for decision boundary animation')
    args = parser.parse_args()

    data = DataSet.from_tsv(args.datafile)
    percep = BinaryPerceptron(dim=data.dim)
    plot = DataPlot(data)
    for i in range(args.epochs):
        print(i, percep.weights, percep.bias)
        rate = print_accuracy(percep, data)
        plot.record_epoch(percep, rate)
        epoch(percep, data, args.alpha)
    print(args.epochs, percep.weights, percep.bias)
    rate = print_accuracy(percep, data)
    plot.record_epoch(percep, rate)
    if args.video:
        plot.animate(args.video)
