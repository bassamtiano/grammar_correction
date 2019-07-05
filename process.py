import os
import sys
import random
import math, copy, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchtext import data, datasets
import spacy

from model import Iterator, BuildModel, rebatch, batch_size_fn, greedy_decode, get_emb

def main():
    src_dir = "data/src"
    model_dir = "data/model"
    eval_dir = "data/eval"

    corpus = "lang8_small"

    en_emb = "glove"
    de_emb = "glove"

    seq_train = False

    emb_dim = 200
    batch_size = 1500

    # Data Loading
    vocab_file = os.path.join(model_dir, "%s.vocab" % (corpus))
    model_file = os.path.join(model_dir, "%s.%s.%s.transformer.pt" % (corpus, en_emb, de_emb))

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Computing Unit
    device = torch.device("cpu")

    # Loading Data
    bos_word = '<s>'
    eos_word = '</s>'

    blank_word = '<blank>'
    min_freq = 2

    spacy_en = spacy.load('en')

    def tokenize(text):
        return [ tkn.text for tkn in spacy_en.tokenizer(text)]

    TEXT = data.Field(tokenize=tokenize, init_token = bos_word, eos_token = eos_word, pad_token=blank_word)

    test = datasets.TranslationDataset(path=os.path.join(src_dir, corpus),
            exts=('.test.src', '.test.trg'), fields=(TEXT, TEXT))
    # use the same order as original data
    test_iter = data.Iterator(test, batch_size=batch_size, device=device,
                              sort=False, repeat=False, train=False)

    random_idx = random.randint(0, len(test) - 1)
    print(test[random_idx].src)
    print(test[random_idx].trg)

    # Vocabulary

    TEXT.vocab = torch.load(vocab_file)
    pad_idx = TEXT.vocab.stoi["<blank>"]

    print("Load %s vocabuary; vocab size = %d" % (corpus, len(TEXT.vocab)))

    # Word Embedding

    encoder_emb, decoder_emb = get_emb(en_emb, de_emb, TEXT.vocab, device, d_model=emb_dim)


    # Translation
    model = BuildModel(len(TEXT.vocab), encoder_emb, decoder_emb,
                       d_model=emb_dim).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    print("Predicting %s ..." % (corpus))

    src, trg, pred = [], [], []
    for batch in (rebatch(pad_idx, b) for b in test_iter):
        out = greedy_decode(model, TEXT.vocab, batch.src, batch.src_mask)
        # print("SRC OUT: ", src.shape, out.shape)
        probs = model.generator(out)
        _, prediction = torch.max(probs, dim = -1)

        source = [[TEXT.vocab.itos[word] for word in words[1:]] for words in batch.src]
        target = [[TEXT.vocab.itos[word] for word in words[1:]] for words in batch.trg]
        translation = [[TEXT.vocab.itos[word] for word in words] for words in prediction]

        for i in range(len(translation)):
            src.append(' '.join(source[i]).split('</s>')[0])
            trg.append(' '.join(target[i]).split('</s>')[0])
            pred.append(' '.join(translation[i]).split('</s>')[0])

            # eliminate data with unkonwn words in src trg
            if '<unk>' in src[-1] or '<unk>' in trg[-1]:
                continue

            print("Source:", src[-1])
            print("Target:", trg[-1])
            print("Translation:", pred[-1])
            print()

    prefix = os.path.join(eval_dir, '%s.%s.%s.eval' % (corpus, en_emb, de_emb))
    for sentences, ext in zip([src, trg, pred], ['.src', '.trg', '.pred']):
        with open(prefix + ext, 'w+') as f:
            f.write('\n'.join(sentences))

if __name__ == "__main__":
    main()
