
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
we now to a* search using patterns? 