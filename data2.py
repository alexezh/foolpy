from typing import List

def makeBasic(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      o.append(f"{a} + {b} => {a} #+ {b}");
      o.append(f"{a} - {b} => {a} #- {b}");

  for a in range(0, 9):
    o.append(f"{a} => #{a}");

def makeAbstract(o: List[str]):
  for a in range(0, 9):
    for b in range(0, 9):
      o.append(f"{a} + {b} => number + number");
      o.append(f"{a} - {b} => {a} #- {b}");

  for a in range(0, 9):
    o.append(f"{a} => #{a}");

