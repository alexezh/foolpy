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

class QADataset(Dataset):
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
    
class MaskDataset(Dataset):
    def __init__(self, word2idx, max_length=10):
      self.questions = []
      self.answers = []
      self.word2idx = word2idx
      self.max_length = max_length

    @staticmethod
    def makeMask(sentenses, word2idx, max_length=10): 
      self = MaskDataset(word2idx, max_length);
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
      self = QADataset(word2idx, max_length);
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
        input_len = len(input)
        input = input[:self.max_length] + [0] * (self.max_length - len(input))
        target = target[:self.max_length] + [0] * (self.max_length - len(target))

        return torch.tensor(input, dtype=torch.long), torch.tensor(target, dtype=torch.long), torch.tensor([input_len]), torch.tensor([target.count(1)])

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.categories = []

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
        self.add_word('0', 'number');
        self.add_word('1', 'number');
        self.add_word('2', 'number');
        self.add_word('3', 'number');
        self.add_word('4', 'number');
        self.add_word('5', 'number');
        self.add_word('6', 'number');
        self.add_word('7', 'number');
        self.add_word('8', 'number');
        self.add_word('9', 'number');
        self.add_word('a', 'variable');
        self.add_word('b', 'variable');
        self.add_word('c', 'variable');
        self.add_word('d', 'variable');
        self.add_word('e', 'variable');        

    def add_word(self, word, category = None):
        if word not in self.word2idx:
            self.idx2word.append(word)
            wordIdx = len(self.idx2word) - 1
            self.word2idx[word] = wordIdx

            if(category != None):
              catIdx = self.word2idx[category]
              if len(self.categories) <= wordIdx:
                self.categories.extend([0] * (wordIdx + 1 - len(self.categories)))
              self.categories[wordIdx] = catIdx

        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

dictionary = Dictionary();

def randNum(max, other):
  while True:
    val = random.randint(0, max)
    if val != other:
      return val

class Corpus2(object):
    def __init__(self, max_length):
        self.dictionary = dictionary

        basic = []
        makeBasic(basic);
  
        self.train = MaskDataset.makeMask(train, self.dictionary.word2idx, max_length)

class Corpus(object):
    def __init__(self, max_length):
        self.dictionary = dictionary
      
        basic = []
        makeBasic(basic);

        full = [];
        makeSums(full)

        random.seed(a=42)
        train = random.sample(full, math.floor(len(full) * 0.8))  # Randomly select 3 elements
        train.extend(basic);
  
        full.extend(basic);
        test = random.sample(full, math.floor(len(full) * 0.20))  # Randomly select 3 elements

        self.train = MaskDataset.makeMask(train, self.dictionary.word2idx, max_length)
        self.test = MaskDataset.makeMask(test, self.dictionary.word2idx, max_length)

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

def emb_hot(indices, num_classes):
    # Create a tensor of shape (N, num_classes) filled with zeros
    one_hot_tensor = torch.zeros(indices.size(0), indices.size(1), num_classes, device=indices.device)
    
    # Set the positions indicated by indices to 1
    one_hot_tensor.scatter_(1, indices.unsqueeze(1), 1)
    
    index_map = torch.tensor(dictionary.categories).to(indices.device)
    cat_indices = index_map[indices]
    one_hot_tensor.scatter_(1, cat_indices.unsqueeze(1), 1)

    return one_hot_tensor