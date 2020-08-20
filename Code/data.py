import torch
from torchtext.vocab import Vectors, GloVe
from torchtext import data, datasets


def load_TREC_data(batch_size= 32, embedding_length = 100, fix_length = 10):
  # set up fields
  tokenize = lambda x: x.split()
  TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length = fix_length)
  # LABEL = data.LabelField()
  LABEL = data.LabelField(dtype=torch.float)

  # make splits for data
  train, test = datasets.TREC.splits(TEXT, LABEL)

  # build the vocabulary
  TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=embedding_length))
  LABEL.build_vocab(train)

  # make iterator for splits
  train_iter, test_iter = data.BucketIterator.splits(
      (train, test), batch_size= batch_size, device=0)
  
  word_embeddings = TEXT.vocab.vectors
  vocab_size = len(TEXT.vocab)

  return TEXT, vocab_size, word_embeddings, train_iter, test_iter