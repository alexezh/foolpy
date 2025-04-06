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
  for v in range(0, 9):
    o.append(f"{v} is {v}");

def makeIsNumber(o: List[str]):
  for v in range(0, 100):
    o.append(f"{v} is number");
  for v in range(0, 100):
    o.append(f"{v} is {v}");

def makeSumDigit(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      o.append(f"{a} + {b} is {a + b}");

def makeBasicVar(o: List[str]):
  for a in range(0, 9):
    o.append(f"a + {a} is a + {a}");
    o.append(f"b + {a} is b + {a}");
    o.append(f"c + {a} is c + {a}");
  o.append(f"a + b is b + a");
  o.append(f"a + b is a + b");
  o.append(f"c + a is c + a");
  o.append(f"c + a is a + c");

def randNum(max, other):
  while True:
    val = random.randint(0, max)
    if val != other:
      return val

def makeSumDigitWrong(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      for n in range(0, 10):
        o.append(f"{a} + {b} is {randNum(9, a + b)} wrong {a} + {b} is {a + b}");
        break

def makeMinDigit(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      if a >= b:
        o.append(f"{a} - {b} is {a - b}");

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
         
def makeSumNumber2(o: List[str]):
  for a in range(0, 99):
    for b in range(0, 99):
      if a + b > 99 or (a < 10 and b < 10): 
         continue
      o.append(f"{a} + {b} is { a + b }");

def makeSumVar2(o: List[str]):
  for a in range(0, 99):
    for b in range(0, 99):
      o.append(f"a + {a} + {b} is a + { a + b }");
      o.append(f"b + {a} + {b} is b + { a + b }");
      o.append(f"c + {a} + {b} is c + { a + b }");
      o.append(f"{a} + a + {b} is a + { a + b }");
      o.append(f"{a} + b + {b} is b + { a + b }");
      o.append(f"{a} + c + {b} is c + { a + b }");

def makeSumDigit3(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      for c in range(0, 9):
        o.append(f"{a} + {b} + {c} is {a + b} + {c} is { a + b + c }");
        o.append(f"{a} + {b} + {c} is {a} + {b + c} is { a + b + c }");
        o.append(f"a + {a} + {b} + {c} is {a} + {b + c} is a + { a + b + c }");
        o.append(f"b + {a} + {b} + {c} is {a} + {b + c} is b + { a + b + c }");
        o.append(f"c + {a} + {b} + {c} is {a} + {b + c} is c + { a + b + c }");
        o.append(f"a + {a} + {b} + c + {c} is {a} + {b + c} is a + c + { a + b + c }");
        o.append(f"b + {a} + {b} + a + {c} is {a} + {b + c} is a + b + { a + b + c }");
        o.append(f"c + {a} + {b} + b + {c} is {a} + {b + c} is c + b + { a + b + c }");

def makeSumDigit4(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      for c in range(0, 9):
        for d in range(0, 9):
          o.append(f"{a} + {b} + {c} + {d} is {a + b} + {c} + {d} is {a + b + c} + {d} is { a + b + c + d }");
          o.append(f"{a} + {b} + {c} + {d} is {a} + {b} + {c + d} is {a} + {b + c + d} is { a + b + c + d }");
          o.append(f"{a} + {b} + {c} + {d} is {a} + {b + c} + {d} is {a} + {b + c + d} is { a + b + c + d }");

          # o.append(f"{a} + {b} + {c} + {d} is {a + b} + {c} + {d} is {a + b + c} + {d} is {randNum(9, a + b + c + d)} is { a + b + c + d }");
          # o.append(f"{a} + {b} + {c} + {d} is {a} + {b} + {c + d} is {a} + {b + c + d} is {randNum(9, a + b + c + d)} is { a + b + c + d }");
          # o.append(f"{a} + {b} + {c} + {d} is {a} + {b + c} + {d} is {a} + {b + c + d} is {randNum(9, a + b + c + d)} is { a + b + c + d }");

          # o.append(f"{a} + {b} + {c} + {d} is {randNum(9, a + b)} + {c} + {d} is {a + b} + {c} + {d} is {a + b + c} + {d} is { a + b + c + d }");
          # o.append(f"{a} + {b} + {c} + {d} is {a} + {b} + {randNum(9, c + d)} wrong is {a} + {b} + {c + d} is {a} + {b + c + d} is { a + b + c + d }");
          # o.append(f"{a} + {b} + {c} + {d} is {a} + {randNum(9, b + c)} + {d} wrong is {a} + {b + c} + {d} is {a} + {b + c + d} is { a + b + c + d }");

def makeSumDigit5(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      for c in range(0, 9):
        for d in range(0, 9):
          for e in range(0, 9):
            o.append(f"{a} + {b} + {c} + {d} + {e} is {a + b} + {c} + {d} + {e} is {a + b + c} + {d} + {e} is { a + b + c + d + e }");
            o.append(f"{a} + {b} + {c} + {d} + {e} is {a} + {b + c} + {d} + {e} is {a + b + c} + {d} + {e} is { a + b + c + d + e }");
            o.append(f"{a} + {b} + {c} + {d} + {e} is {a} + {b + c} + {d} + {e} is {a} + {b + c + d + e} is { a + b + c + d + e }");
            o.append(f"{a} + {b} + {c} + {d} + {e} is {a} + {b} + {c + d} + {e} is {a} + {b + c + d + e} is { a + b + c + d + e }");
            o.append(f"{a} + {b} + {c} + {d} + {e} is {a} + {b} + {c} + {d + e} is {a} + {b + c + d + e} is { a + b + c + d + e }");

class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        
        self.dictionary.add_word('<eos>');
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
        makeIsDigit(basic)
        makeIsNumber(basic)
        makeSumDigit(basic)
        # makeSumDigitWrong(basic)
        makeBasicVar(basic);

        full = [];
        makeSumVar2(full)
        makeSumNumber(full)
        makeSumNumber2(full)
        makeSumDigit3(full);
        makeSumDigit4(full)
        makeSumDigit5(full);

        random.seed(a=42)
        train = random.sample(full, math.floor(len(full) * 0.7))  # Randomly select 3 elements
        train.extend(basic);

        valid = random.sample(full, math.floor(len(full) * 0.15))  # Randomly select 3 elements
        valid.extend(basic);
        
        test = random.sample(full, math.floor(len(full) * 0.20))  # Randomly select 3 elements
        test.extend(basic);

        self.train = self.tokenize(train)
        self.valid = self.tokenize(valid)
        self.test = self.tokenize(test)

    def tokenize(self, exp: List[str]):
      """Tokenizes a text file."""

      # Tokenize file content
      idss = []
      for line in exp:
          ids = []
          words = line.split()
          for word in words:
            idx = self.dictionary.word2idx.get(word);
            if idx == None:
             for c in word:
                ids.append(self.dictionary.word2idx[c])
            else:
               ids.append(idx);
          
          ids.append(self.dictionary.word2idx['<eos>'])
          idss.append(torch.tensor(ids).type(torch.int64))
      ids = torch.cat(idss)

      return ids
