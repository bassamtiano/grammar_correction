import os
import sys
import random
import math, copy, time

# Pretify Print Output
from pprint import pprint

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from torchtext import data, datasets
import spacy

from transformer import MultiHeadedAttention, EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer, Generator
from positioning import PositionwiseFeedForward, PositionalEncoding

# Word Embedding ===============================================================

def get_emb(en_emb_name, de_emb_name, vocab, device, d_model=512):
    emb = nn.Embedding.from_pretrained(vocab.vectors)
    return [emb for name in [en_emb_name, de_emb_name]]

def subsequent_mask(size):
    "Mask out subsequent positions."
    attention_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attention_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# Embedding and Softmax ========================================================

class Embeddings(nn.Module):
    def __init__(self, embedding, d_model):
        super(Embeddings, self).__init__()
        self.lut = embedding
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# Iterator =====================================================================

class Iterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum().item()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


# Make Model ===================================================================

def BuildModel(vocab_size, encoder_emb, decoder_emb, d_model = 512, N = 6, d_ff = 2048, h = 8, dropout = 0.1):

    target_vocab = vocab_size
    c = copy.deepcopy

    attention = MultiHeadedAttention(h, d_model)
    feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    encoder_layer = EncoderLayer(d_model, c(attention), c(feed_forward), dropout)
    decoder_layer = DecoderLayer(d_model, c(attention), c(attention), c(feed_forward), dropout)

    encoder = Encoder(encoder_layer, N)
    decoder = Decoder(decoder_layer, N)

    model = EncoderDecoder( encoder, decoder,
        nn.Sequential(Embeddings(encoder_emb, d_model), c(position)),
        nn.Sequential(Embeddings(decoder_emb, d_model), c(position)),
        Generator(d_model, target_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# Training =====================================================================

def greedy_decode(model, vocab, src, src_mask, trg=None, max_len=60):
    if trg is not None:
        max_len = trg.size(1)

    sents = [[vocab.stoi["<s>"]] for _ in range(len(src))]
    ys = torch.Tensor(sents).type_as(src.data)

    memory = model.encode(src, src_mask)
    for i in range(max_len):
        if trg is not None:
            ys = trg[:, :i + 1]

        out = model.decode(memory, src_mask, Variable(ys),
                Variable(subsequent_mask(ys.size(1)).type_as(src.data)))

        prob = model.generator(out[:, -1])
        _, next_words = torch.max(prob, dim = -1)
        next_words = next_words.view(-1, 1)

        ys = torch.cat([ys, next_words.type_as(src.data)], dim=1)

    return out

# Epoch Loop ===================================================================

def run_epoch(data_iter, model, loss_compute, vocab, seq_train=False):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        if seq_train:
            out = greedy_decode(model, vocab, batch.src, batch.src_mask, trg=batch.trg)
        else:
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Iteration: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed), end='\t')
            start = time.time()
            tokens = 0

            out = greedy_decode(model, vocab, batch.src[:1], batch.src_mask[:1])
            probs = model.generator(out)
            _, s = torch.max(probs, dim = -1)
            trans = [[vocab.itos[w] for w in words] for words in s]
            print("Translation:", ' '.join(random.choice(trans)).split('</s>')[0][:50])

    return total_loss / total_tokens if total_tokens else 0


# Optimizer ====================================================================

class Optimizer:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return Optimizer(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# Batching =====================================================================

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
