from lisp import exec_
from wordencoder import trainWordEmbedding, loadWordEmbedding, testWord
from device import initDevice



# if sequemce the same, make word collapsing it
# word -> pattern detection
# 
# pattern -> action
# a + b => sum
# a + ? + b => sum
# 
# example 3 + 4 * 5 + 3 
#    => 3 + ? + 3 = 6 + ?
#    => 3 + 4 => failure, multiply first
#    => ? + 4 * 5 + ? => 
#
# already tried this with NN, 
# need language to define failure. 
#  => 3 + 4 => failure, multiply first is instruction which increases weight of 
# multiple pattern. So if is meta operation amul 
# we are back to language where parameters are more flexible than normal language
# the sequence is just actions with parameters discovered in context
# 
# the overall sequence is -> choice => sequence. Seqeunce defines a* path
# if we diverge from path, we might go back to choice
# 
# operations can be either one or + which is semi greedy; on each step we might chhoise  different
# decision based on some parameter
#
class Action(object):
  def __init__(self):
    self.id = "hhh"

# action makes execution plan which is probabilistic in nature

initDevice()

# trainWordEmbedding();
loadWordEmbedding();
testWord("hello");
