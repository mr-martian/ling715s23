#!/usr/bin/env python3
import re 
import numpy as np

test_text = '''where are you?
is she in mexico?
i am in greece.
she is in mexico.
is she in england?
'''

train_text = '''are you still here?
where are you?
he is in mexico.
are you tired?
i am tired.
are you in england?
were you in mexico?
is he in greece?
were you in england?
are you in mexico?
i am in mexico.
are you still in mexico? 
are you in greece again?
she is in england.
he is tired.
'''

def tokenise(s):
    return re.sub('([.?])', ' \g<1>', s).split()

vocab = ['<BOS>', '<EOS>', '<PAD>'] + sorted(set(re.sub('([.?])', ' \g<1>', train_text).split()))

DIM = len(vocab)

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

print(word2idx)

x_list = []
y_list = []
for line in train_text.splitlines():
    tokens = ['<BOS>'] + tokenise(line) + ['<EOS>']
    print(tokens)
    for i, tok in enumerate(tokens):
        if i+2 < len(tokens):
            print(i, tok)
            x = np.zeros(DIM*2)
            x[word2idx[tok]] = 1
            x[DIM+word2idx[tokens[i+1]]] = 1
            x_list.append(x)
            y_list.append(word2idx[tokens[i+2]])

X = np.array(x_list)
Y = np.array(y_list)

print(X.shape)
print(Y.shape)

import computation_graph as CG
X_in = CG.ConstantNode(DIM*2)
E = CG.linear_with_sigmoid(X_in, 4)
H = CG.linear_with_sigmoid(E, 6)
O = CG.linear_with_sigmoid(H, DIM)
trainer = CG.softmax_ce_trainer(X_in, O)

for i in range(1, 11):
    print(f'Epoch {i}')
    trainer.epoch(X, Y, alpha=0.01)
    print(f'  loss: {trainer.losses[-1]}')
