#!/usr/bin/env python3

import numpy as np

class CGNode:
    def __init__(self, dim):
        self.prev = {}
        self.deriv = []
        self.dim = dim
        self.cur_value = None
    def getderiv(self, nextd):
        raise NotImplementedError()
    def passderiv(self, nextd):
        self.deriv += list(self.getderiv(nextd))
    def backprop(self, alpha=0.01):
        for k, v in self.prev.items():
            v.passderiv(self.deriv)
            v.backprop(alpha)
        self.update(alpha)
    def update(self, alpha=0.01):
        raise NotImplementedError()
    def clearval(self):
        self.cur_value = None
    def calculate(self):
        raise NotImplementedError()
    def evaluate(self):
        if self.cur_value is None:
            for k, v in self.prev.items():
                v.calculate()
            self.calculate()
        return self.cur_value

class ConstantNode(CGNode):
    def setval(self, X):
        self.cur_value = X
    def getderiv(self, nextd):
        return []
    def update(self, alpha):
        pass
    def calculate(self):
        pass

class LinearNode(CGNode):
    def __init__(self, innode, dim):
        super().__init__(dim)
        self.prev['input'] = innode
        self.W = np.random.rand(innode.dim, dim)
    def calculate(self):
        X = self.prev['input'].evaluate()
        self.cur_value = np.matmul(X, self.W)
    def getderiv(self, nextd):
        return np.transpose(np.matmul(self.W, np.transpose(nextd)))
    def update(self, alpha=0.01):
        X = self.prev['input'].evaluate()
        dvec = np.transpose(np.array([np.sum(np.array(self.deriv), axis=0)]))
        ovec = np.array([np.sum(self.cur_value, axis=0)])
        tm = (dvec @ ovec)
        self.W -= alpha * tm
        self.deriv = []

class BiasNode(CGNode):
    def __init__(self, innode):
        super().__init__(innode.dim)
        self.prev['input'] = innode
        self.B = np.random.rand(innode.dim)
    def calculate(self):
        X = self.prev['input'].evaluate()
        self.cur_value = X + self.B
    def getderiv(self, nextd):
        return nextd
    def update(self, alpha=0.01):
        self.B -= alpha * np.sum(np.array(self.deriv), axis=0)
        self.deriv = []

class SigmoidNode(CGNode):
    def __init__(self, innode):
        super().__init__(innode.dim)
        self.prev['input'] = innode
    def sigma(self, x):
        return np.reciprocal(np.exp(-x)+1)
    def calculate(self):
        X = self.prev['input'].evaluate()
        self.cur_value = self.sigma(X)
    def getderiv(self, nextd):
        sd = self.sigma(np.array(nextd))
        return sd * (1 - sd)
    def update(self, alpha=0.01):
        self.deriv = []

class SoftmaxOutputNode(CGNode):
    def __init__(self, innode):
        super().__init__(innode.dim)
        self.prev['input'] = innode
    def calculate(self):
        X = self.prev['input'].evaluate()
        expx = np.exp(X)
        sumexpz = np.sum(expx, axis=1)
        sigma = np.transpose(expx) / sumexpx
        self.cur_value = np.transpose(sigma)
    def getderiv(self, nextd):
        raise NotImplementedError() # something annoying
    def update(self, alpha=0.01):
        self.deriv = []

class SoftmaxCELossNode(CGNode):
    def __init__(self, innode, refnode):
        super().__init__(1)
        self.prev['input'] = innode
        self.prev['ref'] = refnode
    def calculate(self):
        X = self.prev['input'].evaluate()
        Y = self.prev['ref'].evaluate()
        self.cur_expx = np.exp(X)
        self.cur_sumexpx = np.sum(self.cur_expx, axis=1)
        sigma = np.transpose(self.cur_expx) / self.cur_sumexpx
        self.cur_softmax = np.transpose(sigma)
        self.cur_value = np.array([-np.log(y_hat[y]) for y_hat, y in zip(self.cur_softmax, Y)])
    def getderiv(self, nextd):
        X = self.prev['input'].evaluate()
        Y = self.prev['ref'].evaluate()
        ret = np.zeros((len(Y), self.prev['input'].dim))
        for i in range(len(Y)):
            y = Y[i]
            ex = self.cur_expx[i,y]
            c = self.cur_sumexpx[i]
            ret[i,y] = - (c-ex) / c # Wolfram|Alpha said so
        return np.array(ret)
    def update(self, alpha=0.01):
        self.deriv = []

def linear_with_activation(node, outdim, activ):
    w = LinearNode(node, outdim)
    b = BiasNode(w)
    return activ(b)

def linear_with_sigmoid(node, outdim):
    return linear_with_activation(node, outdim, SigmoidNode)

class Trainer:
    def __init__(self, innode, outnode, lossinnode, lossnode):
        self.innode = innode
        self.outnode = outnode
        self.lossinnode = lossinnode
        self.lossnode = lossnode
        self.losses = []
    def epoch(self, X, Y, alpha=0.01):
        self.innode.setval(X)
        self.lossinnode.setval(Y)
        loss = self.lossnode.evaluate()
        self.losses.append(np.sum(loss) / len(X))
        self.lossnode.deriv = self.lossnode.getderiv(None)
        self.lossnode.backprop(alpha)

def softmax_ce_trainer(innode, outnode):
    softout = SoftmaxOutputNode(outnode)
    const = ConstantNode(1)
    loss = SoftmaxCELossNode(outnode, const)
    return Trainer(innode, softout, const, loss)
