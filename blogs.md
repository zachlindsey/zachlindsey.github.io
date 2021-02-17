---
title: Math & Code Blog
layout: default
use_math: true
---

# The Satisfiability Threshold Conjecture

I recently learned about this conjecture that still remains open, despite being a fairly basic observation. It's about random CNF SAT problems, so let's introduce that. CNF stands for *conjuctive normal form*, and is a restricted form of logical sentences. A sentence is in CNF is it of the form 

\[[C_1 \wedge C_2 \wedge \ldots \wedge C_n,\]]

where each $C_i$ is a *clause*. A clause has the form $l_1 \vee l_2 \ldots \vee l_m$, where each $l_i$ is a *literal*, which can be either a simple propositional variable $x_i$ or its negation $\neg x_i$.

It's not hard to work out that any sentence in propositional logic (only the most basic logical operations: and, or, not, implies and atomic proposition symbols) can be written in this form, and so this makes for a very clean way to boil down complicated statements to a simple format. We can now think about random sentences in CNF. Let $\mbox{CNF}_k(m,n)$ denote the sentences in CNF with $m$ clauses with exactly $k$ literals chosen from $n$ propositional symbols. We can create "random" sentences by drawing uniformly without replacement from this set. As an example, if $k = 3, m = 4, n = 5$, we can form a clause out of the symbols P, Q, R, S, T. We need to make 4 random ones with 3 symbols each, so one might look like...

* P or R or T, and...
* not Q or P or R, and...
* T or S or P, and...
* not R or S or P

For fun, try working out whether or not you can assign TRUE/FALSE to each to make all four statements true.

And now the question: What is the probability that a sentence chosen from $\mbox{CNF}_k(m,n)$ is solvable? That is, is there an assignment of true or false to the variables that makes all the clauses true? Can we estimate how long it will take to solve?

## Experiments

Using [PySat](https://pysathq.github.io/), a python library containing solvers for these SAT problems, we can quickly draw many random sentences and solve them. Let's see how long it takes to solve them and the probability of solvability for various values of $\frac{m}{n}$, the ratio of clauses to symbols. For each of these, let's fix $k = 3$.

Here is the code that I used to generate a plot that examines the questions about CNF-SATs posed above. It takes a little bit to run, so be patient!


```
from pysat.solvers import Glucose3
import numpy as np
import matplotlib.pyplot as plt
import random
import time


num_symb = 50
k = 3


def random_CSP(num_symb, num_clause, k):
    sentence = set()
    
    while len(sentence) < num_clause:
        literals = random.sample(range(1,num_symb+1), k)
        
        for i in range(k):
            if np.random.rand() < 0.5:
                literals[i] *= -1
        literals = tuple(literals)
        
        if literals not in sentence:
            sentence.add(literals)
    return sentence

number_solvable = [[],[],[]]
solve_times = [[],[],[]]
num_trials = 100
scale = 10
for i, num_symb in enumerate([20,50,100]):
    for r in range(1,9*scale):
        number_solvable[i].append(0)
        solve_times[i].append(0)
        num_clause = (num_symb//scale)*r
        
        for _ in range(num_trials):
            CSP = random_CSP(num_symb, num_clause, k)
            g = Glucose3()
            for clause in CSP:
                g.add_clause(clause)
                
            start = time.time()
            result = g.solve()
            stop = time.time()
            g.delete()
            
            solve_times[i][-1] += stop - start
            if result is True:
                number_solvable[i][-1] += 1
         
X = [i/scale for i in range(1,9*scale)]
for Y in solve_times:
    plt.plot(X, Y)
plt.legend([20,50,100], title='n')
plt.xticks(range(1,9))
plt.xlabel('m/n')
plt.ylabel('total run time')
plt.title('Time to Solve CNF-SAT Problems')
plt.show()

for Y in number_solvable:
    plt.plot(X, Y)
plt.legend([20,50,100], title='n')
plt.xticks(range(1,9))
plt.xlabel('m/n')
plt.ylabel(f'number solvable otu of {num_trials}')
plt.title('Probability CNF is Solvable')
plt.show()
```

and the results!

![Solve Times](/images/solve_times.png)
![Solve Probability](/images/solve_probs.png)

Note the *sharp* drop in in whether or not a problem is solvable that happens around $m/n = 4.5$. Intuitively, problems to the left of that cliff have few clauses, but many variables. This means that they tend to be *underconstrained*, and so guessing a solution is quite easy! On the other hand, problems on the right side of the cliff have many constraints but few variables. So they are *overconstrained*, and our SAT solver can pretty quickly discover some contradiction that throws out the possibility of a solution.

The solve times support this conclusion. To the left, we see the solver can quickly guess a solution or fiddle with a guess to find a solution. Towards the right, the solver takes a little longer to realize the constraints cannot be satisfied, but this still happens relatively quickly. Near the cliff, however, we see that the problems are much, much harder, and the solver needs considerably more time to come up with the answer.

What's more interesting is that as the size of the problem grows, the cliff becomes sharper. And now we arrive at the conjecture:

For each $k \geq 2$, there is a cutoff $r_k$ so that, as $n \rightarrow \infty$, a random $\mbox{CNF}_k(m,n)$ problem is solvable with probability 1 if $\frac{m}{n} < r_k$ and not solvable with probability 0 if $\frac{m}{n} > r_k$.

It still seems to be unknown for $k=3$! Proofs for large $k$ rely on methods from statistical physics, and it seems like many other "random" NP-complete problems have similar interesting "phase transition" properties like this. 






