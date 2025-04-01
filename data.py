import os
from io import open
import torch
from torch.utils.data import Dataset, DataLoader
import math
import random
from typing import List

class TokenizedDataset(Dataset):
    def __init__(self, sentenses, word2idx, max_length=10):
      self.content = []
      self.word2idx = word2idx
      self.max_length = max_length

      for sentense in sentenses:
        ids = []
        words = sentense.split()
        for word in words:
          idx = self.word2idx.get(word);
          if idx == None:
            for c in word:
              ids.append(self.word2idx[c])
          else:
              ids.append(idx);
        
        ids.append(self.word2idx['<eos>'])
        self.content.append(ids)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        # Pad or truncate to max_length
        input = self.content[idx]
        target = input[1 : len(input)-1] 
        
        input = input[:self.max_length] + [0] * (self.max_length - len(input))
        target = target[:self.max_length] + [0] * (self.max_length - len(target))

        return torch.tensor(input, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
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

def randNum(max, other):
  while True:
    val = random.randint(0, max)
    if val != other:
      return val

vars = ['a', 'b', 'c'];

def makeBasic(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      o.append(f"{a} + {b} => #{a} #+ #{b}");
      o.append(f"{a} - {b} => #{a} #- #{b}");

  for a in range(0, 9):
    o.append(f"{a} => none");

  for v1 in vars:
    for v2 in vars:
      if v1 != v2:
        o.append(f"{v1} + {v2} => none");
        o.append(f"{v1} - {v2} => none");
      else:
        o.append(f"{v1} + {v2} => none");
        o.append(f"{v1} - {v2} => 0");

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
         
def makeSums(o: List[str]):
  for a in range(0, 99):
    for b in range(0, 99):
      if a + b > 99 or (a < 10 and b < 10): 
         continue
      o.append(f"{a} + {b} => #{a} #+ #{b}");
      o.append(f"{a} + {b} => #{a} #- #{b}");

  for a in range(0, 99):
    for b in range(0, 99):
      for v in vars:
        o.append(f"{v} + {a} + {b} => {v} + #{a} #+ #{b}");
        o.append(f"{a} + {v} + {b} => #{a} + {v} #+ #{b}");
        o.append(f"{a} + {b} + {v} => #{a} #+ #{b} + {v}");

  for a in range(0, 30):
    for b in range(0, 30):
      for c in range(0, 30):
        o.append(f"{a} + {b} + {c} => #{a} #+ #{b} + {c}");
        o.append(f"{a} + {b} + {c} => {a} + #{b} #+ #{c}");

  for a in range(0, 30):
    for b in range(0, 30):
      for c in range(0, 30):
        for v in vars:
          o.append(f"{v} + {a} + {b} + {c} => {v} + {a} + #{b} #+ #{c}");
          o.append(f"{v} + {a} + {b} + {c} => {v} + #{a} + #{b} + {c}");
          o.append(f"{a} + {v} + {b} + {c} => {a} + {v} + #{b} #+ #{c}");
          o.append(f"{a} + {v} + {b} + {c} => #{a} + {v} + {b} #+ #{c}");

  for a in range(0, 20):
    for b in range(0, 20):
      for c in range(0, 20):
        for d in range(0, 20):
          o.append(f"{a} + {b} + {c} + {d} => #{a} + #{b} + {c} + {d}");
          o.append(f"{a} + {b} + {c} + {d} => {a} + {b} + #{c} #+ #{d}");
          o.append(f"{a} + {b} + {c} + {d} => {a} + #{b} #+ #{c} + {d}");


  for a in range(0, 9):
    for b in range(0, 9):
      for c in range(0, 9):
        for d in range(0, 9):
          for e in range(0, 9):
            o.append(f"{a} + {b} + {c} + {d} + {e} => #{a} #+ #{b} + {c} + {d} + {e}");
            o.append(f"{a} + {b} + {c} + {d} + {e} => {a} + #{b} #+ #{c} + {d} + {e}");
            o.append(f"{a} + {b} + {c} + {d} + {e} => {a} + {b} + #{c} #+ #{d} + {e}");
            o.append(f"{a} + {b} + {c} + {d} + {e} => {a} + {b} + {c} + #{d} #+ #{e}");
            o.append(f"{a} + {b} + {c} + {d} + {e} => #{a} + {b} + {c} + {d} #+ #{e}");
            o.append(f"{a} + {b} + {c} + {d} + {e} => #{a} + {b} + {c} #+ #{d} + {e}");

class Corpus(object):
    def __init__(self, max_length):
        self.dictionary = Dictionary()
        
        self.dictionary.add_word('<eos>');
        self.dictionary.add_word('#');
        self.dictionary.add_word('=>');
        self.dictionary.add_word('none');
        self.dictionary.add_word('number');
        self.dictionary.add_word('digit');
        self.dictionary.add_word('wrong');
        self.dictionary.add_word('is');
        self.dictionary.add_word('+');
        self.dictionary.add_word('-');
        self.dictionary.add_word('0');
        self.dictionary.add_word('1');
        self.dictionary.add_word('2');
        self.dictionary.add_word('3');
        self.dictionary.add_word('4');
        self.dictionary.add_word('5');
        self.dictionary.add_word('6');
        self.dictionary.add_word('7');
        self.dictionary.add_word('8');
        self.dictionary.add_word('9');
        self.dictionary.add_word('a');
        self.dictionary.add_word('b');
        self.dictionary.add_word('c');

        basic = []
        makeBasic(basic);

        full = [];
        makeSums(full)

        random.seed(a=42)
        train = random.sample(full, math.floor(len(full) * 0.7))  # Randomly select 3 elements
        train.extend(basic);
        random.shuffle(train)
  
        full.extend(basic);
        valid = random.sample(full, math.floor(len(full) * 0.15))  # Randomly select 3 elements

        test = random.sample(full, math.floor(len(full) * 0.20))  # Randomly select 3 elements

        self.train = TokenizedDataset(train, self.dictionary.word2idx, max_length)
        self.valid = TokenizedDataset(valid, self.dictionary.word2idx, max_length)
        self.test = TokenizedDataset(test, self.dictionary.word2idx, max_length)

    def tokenize(self, str):
      ids = []
      words = str.split()
      for word in words:
        idx = self.dictionary.word2idx.get(word);
        if idx == None:
          for c in word:
            ids.append(self.word2idx[c])
        else:
            ids.append(idx);
      return idx