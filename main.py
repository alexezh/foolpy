from lisp import exec_
from wordencoder import trainWordEmbedding, loadWordEmbedding, testWord
from device import initDevice

exec_("""(for-each 
  (lambda (o)
    (if 
      (mmatch o)
      (maction o)
    )
  ) objects
)""")

initDevice()

# trainWordEmbedding();
loadWordEmbedding();
testWord("hello");
