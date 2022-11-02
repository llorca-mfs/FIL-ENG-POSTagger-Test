import torch
import torch.nn as nn
from collections import Counter

# Normalizes quotes and lowercases a string
def normalize(line): return line.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").lower()

# Indexes, truncates, then pads
def proc_set(s, word2idx, word_vocab, msl=128):
    proc = []
    for line in s:
        line = [word2idx[w if w in word_vocab else '<unk>'] for w in line.split()][:msl]
        if len(line) < msl: line += [word2idx['<pad>'] for _ in range(msl - len(line))]
        proc.append(line)
    return proc

def produce_vocab(words, min_freq=2):
    vocab = []
    for line in words: vocab.extend(line.split())
    counts = dict(Counter(vocab))
    vocab = [word for word in counts.keys() if counts[word] >= min_freq]
    vocab = set(vocab + ['<unk>', '<pad>'])
    idx2word = list(vocab)
    word2idx = {idx2word[i]:i for i in range(len(idx2word))}
    return vocab, idx2word, word2idx

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)

# Categorical accuracy
def accuracy(out, y, tag2idx):
    msl, bs = y.shape
    acc = 0
    with torch.no_grad():
        preds = out.argmax(2)
        for i in range(bs):
            nz = (y[:,i] != tag2idx['<pad>']).sum().item()
            acc += (preds[:,i][:nz] == y[:,i][:nz]).float().mean().item()
    acc /= bs
    return acc

# Process one example string then returns a prediction
def predict(s, word2idx, idx2tag, word_vocab, msl, model):
    # Convert to input
    xs = normalize(s)
    l = len(xs.split())
    xs = proc_set([xs], word2idx, word_vocab, msl=msl)
    xs = torch.LongTensor(xs).transpose(1, 0)
    
    # Produce prediction
    with torch.no_grad():
        out = model(xs)
    out = out.squeeze(1).argmax(1)
    preds = list(out[:l].numpy())

    preds = [idx2tag[ix] for ix in preds]
    return preds
