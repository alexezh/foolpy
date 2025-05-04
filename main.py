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

# solve 2 + x - (2 + y) + x + 4
#  group_subexp - map()
#  simplify
#  ungroup_subexp
#    why would we open - to continue, so we need goal
#    to compute with X, I either need to open subexp, or bring into
#    we can just compute in parallel for couple steps and compare the results
#    later we can "analyze" what happened and decide why. Such as that we want to cut x
#  criteriea - smaller + !same + 
# solve then looks as following
#   second part is goal, we want to be number, or number + var
#   but for this we need to know which vars we have, so group objects, lambda(x, x => "number")
#   and then we check that each category used once
# simplify
result = exec_("""
(msolve 
  (lambda (o)
    (if 
      (mmatch o)
      (maction o)
    )
  )
  (lambda ()
    (more
      (reduce 
        (group objects 
          (lambda (o) 
            (if 
              (isnumber o)
              "number"
              o
            )
          )
        )
        (lambda (group)
          (if
            (more 
              (len group)
              1
            )
            1
            0
          )
        )
      )
      0
    )
  )
)""")

initDevice()

# trainWordEmbedding();
loadWordEmbedding();
testWord("hello");
