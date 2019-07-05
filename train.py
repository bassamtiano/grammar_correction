import os
import sys
import random
import argparse
import math, copy, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchtext import data, datasets
import spacy

from model import Iterator, Optimizer
from model import BuildModel, run_epoch, batch_size_fn, get_emb, greedy_decode, rebatch

from regularization import LabelSmoothing, LossCompute

def main():

    src_dir = "data/src"
    model_dir = "data/model"
    corpus = "lang8_small"

    en_emb = "glove"
    de_emb = "glove"

    seq_train = False

    emb_dim = 200

    batch_size = 500
    epoches = 100

    # Data Loading
    vocab_file = os.path.join(model_dir, "%s.vocab" % (corpus))
    model_file = os.path.join(model_dir, "%s.%s.%s.transformer.pt" % (corpus, en_emb, de_emb))

    # Options

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Computing Unit
    device = torch.device("cpu")

    # Data Loading
    bos_word = '<s>'
    eos_word = '</s>'

    blank_word = '<blank>'
    min_freq = 2

    spacy_en = spacy.load('en')

    def tokenize(text):
        return [ tkn.text for tkn in spacy_en.tokenizer(text)]

    TEXT = data.Field(tokenize= tokenize, init_token = bos_word, eos_token = eos_word, pad_token=blank_word)

    train = datasets.TranslationDataset(path=os.path.join(src_dir, corpus),
            exts=('.train.src', '.train.trg'), fields=(TEXT, TEXT))
    val = datasets.TranslationDataset(path=os.path.join(src_dir, corpus),
            exts=('.val.src', '.val.trg'), fields=(TEXT, TEXT))


    train_iter = Iterator(train, batch_size=batch_size, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = Iterator(val, batch_size=batch_size, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

    random_idx = random.randint(0, len(train) - 1)
    print(train[random_idx].src)
    print(train[random_idx].trg)


    # Costruct Vocabulary

    if os.path.exists(vocab_file):
        TEXT.vocab = torch.load(vocab_file)
    else:
        print("Save %s vocabuary..." % (corpus), end='\t')
        TEXT.build_vocab(train.src, min_freq=min_freq, vectors='glove.6B.200d')
        print("vocab size = %d" % (len(TEXT.vocab)))
        torch.save(TEXT.vocab, vocab_file)

    pad_idx = TEXT.vocab.stoi["<blank>"]

    # Word Embedding
    encoder_emb, decoder_emb = get_emb(en_emb, de_emb, TEXT.vocab, device, d_model=emb_dim)

    # Training
    model = BuildModel(len(TEXT.vocab), encoder_emb, decoder_emb, d_model=emb_dim).to(device)
    if os.path.exists(model_file):
        print("Restart from last checkpoint...")
        model.load_state_dict(torch.load(model_file))

    criterion = LabelSmoothing(size=len(TEXT.vocab), padding_idx=pad_idx, smoothing=0.1).to(device)
    model_opt = Optimizer(emb_dim, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0,
                        betas=(0.9, 0.98), eps=1e-9))

    # calculate parameters
    total_params = sum(p.numel() for p in model.parameters()) // 1000000
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) // 1000000
    rate = trainable_params / total_params
    print("Model parameters trainable (%d M) / total (%d M) = %f" % (trainable_params, total_params, rate))

    print("Training %s..." % (corpus))

    for epoch in range(epoches):
        model.train()
        loss_compute = LossCompute(model.generator, criterion, opt=model_opt)
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model, loss_compute, TEXT.vocab, seq_train=seq_train)

        model.eval()
        total_loss, total_tokens = 0, 0
        for batch in (rebatch(pad_idx, b) for b in valid_iter):
            out = greedy_decode(model, TEXT.vocab, batch.src, batch.src_mask, trg=batch.trg)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens

        print("Save model...")
        torch.save(model.state_dict(), model_file)

        print("Epoch %d/%d - Loss: %f" % (epoch + 1, epoches, total_loss / total_tokens))

if __name__ == "__main__":
    main()
