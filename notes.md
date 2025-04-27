
maybe I need to go back to rnn, but again make it in a different way
in markov, we say probability of X|Y. But in this case we want to
fit result into pattern
we say - 1 + 3 * 4 - I know from before to look at 3 * 4 as it was a rule
so how do I encode the rule? We need a way to run sequence on CPU, and then
optimize. So if we just run sequence rule: blah and then we train, we will get
the result. Which means that we need network per-rule. Which we can try training for now

the rule is X, Y, Z

with hot encoding, we limited by indexes. How do I get from hot encode to embedding
goal is to approximate rule. 

I have sum blah - utility function is cost as number of operations
we now to a* search using patterns? We have context, so we select most probable rule
our language boils down to give and take

so if we say that our language is limited to few tokens and the rest is variations, we can define
hierarchical encoding. How ? And then we have 

A, B, C -> X, remember Y
Y, A, B, C, X - C, remember 

hot encode with categories looks interesting. Which is basically embedding in a very sparse space

problem are
- too many tokens
- key store for memory and prediction

for digits, use +1 approach? 9 is 8 + 1, or 9 after 8 which is sequence
then any term is fixed vector with 5-7 terms, and terms refer to other term until basic

which correlates to our memory chain, we remember by blocks of N which popup into head, and this is
the same as dance or movement sequences. 

So we start with basic dim 42 vector with 001 and start building on top. Why 42 - good number

take, 
give,
object,
me,
other,
next


Out context is 7 * 42 vector 294

one is (..objrctt..)

how do we get new 42 vector? need matrix to convert, 
this matrix is memory translation. We can start with one, but it can be multi-layered
so we have 7 * 42 -> 42 and 42 -> 7 * 42

this translation is a key. It cannot be random, as we want to keep math?
two => next one
next of one => two

what if we have set of matrix, initialized by context, making translation from 294 space to 42 which is basically auto-encoder
and 42 -> 294 is rag.

how do I arrive to the same two? 

two is next of one
1 plus one is two
i have two hands
3 is next 2
3 is 2 next of 1

we arrive to the same 3 which means training encoder to arrive there. RAG which can be re-trained, yep? 
