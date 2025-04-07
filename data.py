import os
from io import open
import torch
from torch.utils.data import Dataset, DataLoader
import math
import random
from typing import List

vars = ['a', 'b', 'c', 'd', 'e'];

def makeBasic(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      o.append(f"{a} + {b} => #{a} #+ #{b}");
      o.append(f"{a} - {b} => #{a} #- #{b}");

  for a in range(0, 9):
    o.append(f"{a} => #{a}");

"""   for v1 in vars:
    for v2 in vars:
      if v1 != v2:
        o.append(f"{v1} + {v2} => none");
        o.append(f"{v1} - {v2} => none");
      else:
        o.append(f"{v1} + {v2} => none");
        o.append(f"{v1} - {v2} => 0");
 """
         
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
        o.append(f"{v} - {a} + {b} => {v} - #{a} #+ #{b}");
        o.append(f"{a} + {v} - {b} => #{a} + {v} #- #{b}");
        o.append(f"{a} - {b} - {v} => #{a} #- #{b} - {v}");

  for a in range(0, 30):
    for b in range(0, 30):
      for c in range(0, 30):
        o.append(f"{a} + {b} + {c} => #{a} #+ #{b} + {c}");
        o.append(f"{a} + {b} + {c} => {a} + #{b} #+ #{c}");
        o.append(f"{a} - {b} + {c} => #{a} #- #{b} + {c}");
        o.append(f"{a} + {b} - {c} => {a} + #{b} #- #{c}");
        o.append(f"{a} - {b} - {c} => {a} - #{b} #- #{c}");

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
