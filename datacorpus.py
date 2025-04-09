import os
from io import open
import torch
from torch.utils.data import Dataset, DataLoader
import math
import random
from typing import List

from data import makeBasic, makeSums

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
    def __init__(self, word2idx, max_length=10):
      self.questions = []
      self.answers = []
      self.word2idx = word2idx
      self.max_length = max_length

    def asQuestionAnswer(self, sentenses): 
      for sentense in sentenses:
        parts = sentense.split("=>")
        self.questions.append(tokenizeToIds(parts[0], self.word2idx));
        self.answers.append(tokenizeToIds(parts[1], self.word2idx));

    @staticmethod
    def makeMask(sentenses, word2idx, max_length=10): 
      self = TokenizedDataset(word2idx, max_length);
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
      return self;
    
    @staticmethod
    def makeRel(sentenses, word2idx, max_length=10): 
      self = TokenizedDataset(word2idx, max_length);
      for sentense in sentenses:
        parts = sentense.split("is")
        self.questions.append(tokenizeToIds(parts[0], self.word2idx));
        self.answers.append(tokenizeToIds(parts[1], self.word2idx));
      return self;

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

        self.add_word('<pad>');
        self.add_word('<eos>');
        self.add_word('#');
        self.add_word('=>');
        self.add_word('none');
        self.add_word('quantity');
        self.add_word('symbol');
        self.add_word('countable');
        self.add_word('number');
        self.add_word('digit');
        self.add_word('variable');
        self.add_word('wrong');
        self.add_word('is');
        self.add_word('+');
        self.add_word('-');
        self.add_word('0');
        self.add_word('1');
        self.add_word('2');
        self.add_word('3');
        self.add_word('4');
        self.add_word('5');
        self.add_word('6');
        self.add_word('7');
        self.add_word('8');
        self.add_word('9');
        self.add_word('a');
        self.add_word('b');
        self.add_word('c');
        self.add_word('d');
        self.add_word('e');        

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

dictionary = Dictionary();

def randNum(max, other):
  while True:
    val = random.randint(0, max)
    if val != other:
      return val

class Corpus(object):
    def __init__(self, max_length):
        self.dictionary = dictionary
      
        basic = []
        makeBasic(basic);

        full = [];
        makeSums(full)

        random.seed(a=42)
        train = random.sample(full, math.floor(len(full) * 0.4))  # Randomly select 3 elements
        train.extend(basic);
  
        full.extend(basic);
        test = random.sample(full, math.floor(len(full) * 0.20))  # Randomly select 3 elements

        self.train = TokenizedDataset.makeMask(train, self.dictionary.word2idx, max_length)
        self.test = TokenizedDataset.makeMask(test, self.dictionary.word2idx, max_length)

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
