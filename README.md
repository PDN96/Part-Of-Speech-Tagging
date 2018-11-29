POS Tagging was done using three models:

1) Simple model: It had no dependencies between the tags.
2) HMM model: It had dependencies between the hidden variables (tags) but it was solvable using variable elimination.
3) Complex model: It had complex dependencies between hidden variables and was solved using MCMC sampling.
