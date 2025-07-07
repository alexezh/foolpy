
from random import randint


mm = [1] * 250
visited = [False] * 250

def findPath(x1, y1, x2, y2):
    if(x1 == x2 and y1 == y2):
        return True
    
    if(visited[x1 + y1 * 5]):
        return False
        
    visited[x1 + y1 * 5] = True

    if(mm[x1 + y1 * 5]):
        return False

    if(findPath(x1+1, y1, x2, y2)):
        return True
    if(findPath(x1, y1+1, x2, y2)):
        return True
    if(findPath(x1-1, y1, x2, y2)):
        return True
    if(findPath(x1, y1-1, x2, y2)):
        return True
    
    return False

empty = 0
for i in range(0, 10000):
    idx = randint(0, 250-1)
    mm[idx] = 0 
    empty+=1
    visited = [False] * 250
    if(findPath(0, 1, 4, 4)):
        break;
