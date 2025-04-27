import random
from typing import List
from torch.utils.data import Dataset, DataLoader

import torch


class ConceptDictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

        self.add_word('<pad>');
        self.add_word('<eos>');
        # self.add_word('#');
        # self.add_word('=>');
        # self.add_word('none');
        self.add_word('quantity');
        self.add_word('symbol');
        self.add_word('countable');
        self.add_word('number');
        self.add_word('digit');
        self.add_word('variable');
        self.add_word('wrong');
        # self.add_word('is');
        self.add_word('+');
        self.add_word('-');
        self.add_word('object');
        self.add_word('action');
        self.add_word('1');
        self.add_word('2');
        self.add_word('3');
        self.add_word('many');

        # self.add_word('0', 'number');
        # self.add_word('1', 'number');
        # self.add_word('2', 'number');
        # self.add_word('3', 'number');
        # self.add_word('4', 'number');
        # self.add_word('5', 'number');
        # self.add_word('6', 'number');
        # self.add_word('7', 'number');
        # self.add_word('8', 'number');
        # self.add_word('9', 'number');
        # self.add_word('a', 'variable');
        # self.add_word('b', 'variable');
        # self.add_word('c', 'variable');
        # self.add_word('d', 'variable');
        # self.add_word('e', 'variable');        

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            wordIdx = len(self.idx2word) - 1
            self.word2idx[word] = wordIdx

            # if(category != None):
            #   catIdx = self.word2idx[category]
            #   if len(self.categories) <= wordIdx:
            #     self.categories.extend([0] * (wordIdx + 1 - len(self.categories)))
            #   self.categories[wordIdx] = catIdx

        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Alphabet(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

        self.add_letter('<pad>');
        for idx, letter in enumerate('abcdefghijklmnopqrstuvwxyz'):
          self.add_letter(letter);

    def add_letter(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            wordIdx = len(self.idx2word) - 1
            self.word2idx[word] = wordIdx

            # if(category != None):
            #   catIdx = self.word2idx[category]
            #   if len(self.categories) <= wordIdx:
            #     self.categories.extend([0] * (wordIdx + 1 - len(self.categories)))
            #   self.categories[wordIdx] = catIdx

        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
    
dictionary = ConceptDictionary();
# size of single concept vector
concept_size = 42;
# number of concepts in window
concept_window = 7
max_string = 16
char_to_idx = {ch: idx for idx, ch in enumerate('abcdefghijklmnopqrstuvwxyz')}
string_enc_size = 16 * 26

class WordDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
      return string_to_onehot(self.lines[idx].strip(), max_string).view(-1)
    
# tensor n:26
def string_to_onehot(s, max_size):
    onehot = torch.zeros(max_size, len(char_to_idx))
    for i, ch in enumerate(s):
        if i < 16:
          if ch in char_to_idx:
              onehot[i, char_to_idx[ch]] = 1
    return onehot
    
wordset = WordDataset('eng10000.txt')
wordloader = DataLoader(wordset, batch_size=32, shuffle=True)

def makeConcepts(o: List):
  for idx in range(0, concept_size):
    for a in range(0, concept_window):
      x = torch.zeros(concept_size * concept_window)
      x[a * concept_size + idx] = 1
      o.append(x)
  
  random.shuffle(o)

# different elements on positions
def makeConcepts(o: List):
  for idx in range(0, concept_size):
    for a in range(0, concept_window):
      x = torch.zeros(concept_size * concept_window)
      x[a * concept_size + idx] = 1
      o.append(x)
  
  random.shuffle(o)

def vectorize(exp: List[str], o: List[torch.tensor]):
  for idx in range(0, concept_size):
    for a in range(0, concept_window):
      x = torch.zeros(concept_size * concept_window)
      x[a * concept_size + idx] = 1
      o.append(x)
  
  random.shuffle(o)

# how do I train auto-encoder to use the same output for different query
# as 3+2 = 4+1 = 5. It has to translate into something more basic, like counting apples
# such as 4 is 1111, and it is next 3. What if it is first value, or secomd, does not matter
# 1+3 = 2+2 = next 3 = 4

# each concept is small network which takes different inputs and 
# makes an output. 
def makeDerived(o: List[torch.tensor]):
  exp = [];
  exp.append("1 + 1 => 2")
  exp.append("1 + 1 => 2")
  exp.append("1 + 2 => 3")
  exp.append("2 + 1 => 3")


def tempDerived(o: List):
  o.append("1 => #1")
  o.append("2 => #2")
  o.append("3 => #3")
  o.append("#1 => 1")
  o.append("#2 => 2")
  o.append("#3 => 3")
  o.append(f"#{a} + 1 => #{a + 1}")
  for a in range(4, 10):
    o.append(f"#{a} + 1 => #{a + 1}")

  for a in range(1, 9):
    o.append(f"#{a} => number")
    # with this lookup of 4 will bring next 3 and 3 + 1
    o.append(f"next #{a} => #{a + 1}")

    # o.append(f"#{a} + #2 => #{a + 2}")
    # this goes into memory territory, We can 
    # o.append(f"#{a} + #2 => #{a + 1} + #1")
   
  #  o.append("object object is group of two")
  #  o.append("object is group of one")

  #  o.append("len group one contains one")
  #  o.append("group one contains one")
  #  o.append("len (object) is two")
  #  o.append("len (object objecct) is two")

  # sum of two vars is undefined???
  # sum is function, val(v) is function, sum relate count, quantity
  # but then I need weights for this rels, which we learn
