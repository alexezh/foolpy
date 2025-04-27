import random
from typing import List

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

dictionary = ConceptDictionary();
# size of single concept vector
concept_size = len(dictionary.idx2word);
# number of concepts in window
concept_window = 7

def makeConcepts(o: List):
  for idx in range(0, concept_size):
    for a in range(0, concept_window):
      x = torch.zeros(concept_size * concept_window)
      x[a * concept_size + idx] = 1
      o.append(x)
  
  random.shuffle(o)

  # sum of two vars is undefined???
  # sum is function, val(v) is function, sum relate count, quantity
  # but then I need weights for this rels, which we learn
