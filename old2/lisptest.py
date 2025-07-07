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

# we are building A* algorithm
# which starts with random walk of approaches
# and makes weighted graph based on past experience
# so we are talking about X * W => F. 

# ---- above makes sense, it is question of applying pattern such as 
# treat as variable on a problem based on past experience
# so I need a language which is concatenable? 
# with a*, I need to look at problem as - sequence -> op -> sequence
# where first operations are random.. But I also need way to internalize
# the goal and make sub-goals. This mean making less defined steps in a* as goal
# and going there. But is it really sub-goal, or we have internal pattern X which 
# we apply without having word for it; a pattern can fail which is the same as with
# other patterns. And then we can say - if X works, we then do Y and Z because we 
# learned about such big steps. We then work on reaching X. 

# so the requirement for runtime is that it is fuzzy. 
# each action is meta, and in some cases we just know that it is 100% 

# and this is a* so we have notion of cost which does not let us go 
# to far from the target. 

# input 2x + 5x + 7, goal simplify 
# has parentesis - open or combine
# has divide
# has X, combine
# has numbers - combine
# when we divide and cancel - our cost is low, which is good outcome
# equations - correct form

# goal in mnemonic is
# objects is a number, or ax + by + c, or axy + x + y + c
# we learned it by training as these were good answers
# and we learned canonical form. so match is about breaking things apart
# match(objects, target)
# goal - either number, or variable + number
# https://rdweb.wvd.microsoft.com/api/arm/feeddiscovery