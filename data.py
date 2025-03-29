import os
from io import open
import torch
import math
import random
from typing import List

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def makeIsDigit(o: List[str]):
  for v in range(0, 9):
    o.append(f"{v} is digit");

def makeIsNumber(o: List[str]):
  for v in range(0, 100):
    o.append(f"{v} is number");

def makeSumDigit(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      o.append(f"{a} + {b} is {a + b}");

def makeSumNumber(o: List[str]):
  for a in range(0, 99):
    for b in range(0, 99):
      if a + b > 99 or (a < 10 and b < 10): 
         continue
      if a >= 10 and b < 10:
        a1 = math.floor(a / 10) * 10
        a2 = a - a1;
        o.append(f"{a} + {b} is {a1} + {a2} + {b} is {a1} + {a2 + b} is { a + b }");
      elif a < 10 and b >= 10:
        b1 = math.floor(b / 10) * 10
        b2 = b - b1;
        o.append(f"{a} + {b} is {a} + {b1} + {b2} is {b1} + {a + b2} is { a + b }");
      else:
        a1 = math.floor(a / 10) * 10
        a2 = a - a1;
        b1 = math.floor(b / 10) * 10
        b2 = b - b1;
        o.append(f"{a} + {b} is {a1} + {a2} + {b1} + {b2} is {a1} + {b1} + {a2 + b2} is { a + b }");
         

class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        
        full = []
        makeIsDigit(full)
        makeIsNumber(full)
        makeSumDigit(full)
        makeSumNumber(full)

        train = random.sample(full, math.round(len(full) * 0.7))  # Randomly select 3 elements
        valid = random.sample(full, math.round(len(full) * 0.15))  # Randomly select 3 elements
        test = random.sample(full, math.round(len(full) * 0.20))  # Randomly select 3 elements

        self.train = self.tokenize(train)
        self.valid = self.tokenize(valid)
        self.test = self.tokenize(test)

    def tokenize(self, exp: List[str]):
      """Tokenizes a text file."""
      for line in exp:
          words = line.split() + ['<eos>']
          for word in words:
              self.dictionary.add_word(word)

      # Tokenize file content
      idss = []
      for line in exp:
          words = line.split() + ['<eos>']
          ids = []
          for word in words:
              ids.append(self.dictionary.word2idx[word])
          idss.append(torch.tensor(ids).type(torch.int64))
      ids = torch.cat(idss)

      return ids
