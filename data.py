import os
from io import open
import torch
from torch.utils.data import Dataset, DataLoader
import math
import random
from typing import List

def tokenizeToIds(value, word2idx):
    ids = []
    words = value.split()
    for word in words:
      idx = word2idx.get(word);
      if idx == None:
        for c in word:
          ids.append(word2idx[c])
      else:
          ids.append(idx);
    
    ids.append(word2idx['<eos>'])
    return ids;

def tokenizeToWords(value, word2idx):
    o = []
    words = value.split()
    for word in words:
      idx = word2idx.get(word);
      if idx == None:
        for c in word:
          o.append(c)
      else:
          o.append(word);
    
    return o;

class TokenizedDataset(Dataset):
    def __init__(self, sentenses, word2idx, max_length=10):
      self.questions = []
      self.answers = []
      self.word2idx = word2idx
      self.max_length = max_length

      self.asMask(sentenses);

    def asQuestionAnswer(self, sentenses): 
      for sentense in sentenses:
        parts = sentense.split("=>")
        self.questions.append(tokenizeToIds(parts[0], self.word2idx));
        self.answers.append(tokenizeToIds(parts[1], self.word2idx));

    def asMask(self, sentenses): 
      for sentense in sentenses:
        parts = sentense.split("=>")
        self.questions.append(tokenizeToIds(parts[0], self.word2idx));
        answer = tokenizeToWords(parts[1], self.word2idx);

        mask = []
        i = 0
        while i < len(answer):
          t = answer[i]
          if t == '#':
            i += 2
            mask.append(1)
          else:
            i += 1
            mask.append(0)

        self.answers.append(mask);
    
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # Pad or truncate to max_length
        # input = self.content[idx]
        # target = input[1 : len(input)-1] 
        
        input = self.questions[idx]
        target = self.answers[idx];
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

class Corpus(object):
    def __init__(self, max_length):
        self.dictionary = Dictionary()
        
        self.dictionary.add_word('<pad>');
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
        self.dictionary.add_word('d');
        self.dictionary.add_word('e');

        basic = []
        makeBasic(basic);

        full = [];
        makeSums(full)

        random.seed(a=42)
        train = random.sample(full, math.floor(len(full) * 0.4))  # Randomly select 3 elements
        train.extend(basic);
  
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
      return ids
    

vars = ['a', 'b', 'c', 'd', 'e'];

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
