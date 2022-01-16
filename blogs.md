---
title: Math & Code Blog
layout: default
use_math: true
---

# Old Papers - Dropout

(WIP)

The title of this paper is **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**, by Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov

Dropout should be an easy technique to reproduce with the old datasets, right? It's pretty straightforward to understand and the datasets were smaller back then. Right?

WRONG.

## MNIST

Let's start with MNIST. The authors tested their dropout technique on a 2 to 4 layer net with 1024 to 8096 neurons per layer. My first test with no dropout was a net with 3 hidden layers and 1024 units per layer. I train on the MNIST training data and validate on the MNIST test data, sweeping through various combinations of learning rate, momentum, and batch size. Here is the result, letting the best 10 models run 20 epochs.

![first nodropout sweep](/images/mnist_nodrop_1024x3_sweep.png)

We get error rates between 1.5 and 2%, which is roughly what the paper claims. Also note that these are all converging - classification error is 0 and losses are on the order of 0.0001. Now, let's throw in dropout. In this case, 10 or 20 epochs is not long enough. I ran the best 10 for 40 epochs, and got this.

![first dropout sweep](/images/mnist_drop_1024x3_sweep.png)

That's not good! Do I need to train longer? Do I need to use the max-norm constraint they're talking about? What gives?

(Interesting observation: my top ten models tended to have high momentum paired with low learning rates. This makes sense, since smaller steps paired with more momentum probably leads to a constant "speed" across the loss landscape. Probably one of these speeds is optimal for the data.)

### MORE TRAINING
Let's train one of the dropout models for much, much, much longer.
Doesn't help!

![long minst training with dropout](/images/mnist_dropout_1024x3_longtrain.png)

### BIGGER NETS
Let's try training a larger network. Since dropout is supposed to help deal with overfitting, we might see better results if we move up to a network that's large enough to overfit by a lot. I'll use a network with layers sizes 2048–2048–2048–2048 and see what happens. I did a quick hyperparameter search to find a reasonable configuration for the no dropout network, then ran the best one until convergence. That happens:

```
Epoch 9/99
-----
100%|██████████| 600/600 [01:17<00:00, 7.74it/s]
train Loss: 0.0006
train Error: 0.0000
100%|██████████| 100/100 [00:04<00:00, 21.96it/s]
val Loss: 0.0670
val Error: 0.0173
```

And when I run the network with dropout with the same hyperparameters, FINALLY, I get these for the last few losses:
```tensor(0.0143), tensor(0.0141), tensor(0.0152), tensor(0.0137), tensor(0.0147), tensor(0.0142), tensor(0.0141), tensor(0.0135), tensor(0.0136), tensor(0.0138), tensor(0.0140), tensor(0.0139), tensor(0.0143)```
I count this as my first victory. The takeaway is probably that the network on which you're using dropout out needs to be sufficiently large. This might be because the "dropped out" networks will be too small to be good. It might also be because regularizing something that's not overfit doesn't help.

## Google Street View 

[Kaggle notebook](https://www.kaggle.com/zlindsey/street-view-houses-dropout-vs-no-dropout?scriptVersionId=85254927)

This dataset is a collection of images of house numbers, presumably taken from a car moving past it. Each image has one house number that is centered, and the goal is to predict which digit it is. We're going to more or less copy the architecture described in the paper, as well as any hyperparameters we can find.

> The convolutional layers have 96, 128 and 256 filters respectively. Each convolutional layer has a 5 × 5 receptive field applied with a stride of 1 pixel. Each max pooling layer pools 3 × 3 regions at strides of 2 pixels. The convolutional layers are followed by two fully connected hidden layers having 2048 units each. All units use the rectified linear activation function. Dropout was applied to all the layers of the network with the probability of retaining the unit being p = (0.9, 0.75, 0.75, 0.5, 0.5, 0.5) for the different layers of the network (going from input to convolutional layers to fully connected layers). In addition, the max-norm constraint with c = 4 was used for all the weights. A momentum of 0.95 was used in all the layers

The data is also normalized to have mean 0, variance 1 along each RGB channel. 

### no dropout
For no dropout, we get the following train and val chart.

![street view no dropout](/images/streetview_nodrop.png)

```
Epoch 48/49
----------
100%|██████████| 4579/4579 [00:37<00:00, 122.79it/s]
train Loss: 0.0001
train Error: 0.0000
100%|██████████| 1627/1627 [00:05<00:00, 279.19it/s]
val Loss: 0.6476
val Error: 0.0583
```

### dropout

For dropout...

![street view dropout](/images/streetview_drop.png)

```
Epoch 49/49
----------
100%|██████████| 4579/4579 [00:45<00:00, 100.73it/s]
train Loss: 0.1914
train Error: 0.0572
100%|██████████| 1627/1627 [00:07<00:00, 211.14it/s]
val Loss: 0.2603
val Error: 0.0626
```

*YIKES!* It's worse, again! What gives? The old solution to this problem was to make the net roughly twice the size, so let's try that...

### BIG net, no dropout

### Other tricks - lr decay
The paper mentions starting with initial learning rates around 10 to 0.1, and decaying them by a multilicative factor each epoch. Starting from 10 or 1 seems to get the network "stuck" at a very high loss that never decreases in either the train OR test set.

One setup that seems to work is starting with an LR of about 0.001 and decaying it to 0.00001 over 100 epochs. This network actually converges, even with dropout!

The dropout results:

```
Epoch 99/99
----------
100%|██████████| 4579/4579 [01:35<00:00, 47.83it/s]
train Loss: 0.0261
train Error: 0.0085
100%|██████████| 1627/1627 [00:11<00:00, 143.96it/s]
val Loss: 0.2447
val Error: 0.0501
```

![street view dropout lr decay](/images/streetview_drop_lrdecay.png)

The no dropout results:

```
Epoch 19/99
----------
100%|██████████| 4579/4579 [01:46<00:00, 42.82it/s]
train Loss: 0.0002
train Error: 0.0000
100%|██████████| 1627/1627 [00:15<00:00, 102.59it/s]
val Loss: 0.5244
val Error: 0.0578
```

![street view no drop lr decay](/images/streeview_nodrop_lrdecay.png)



(coming soon!)


## Some other reading
In googling around for trying to understand dropout a little better, I uncovered these papers that I might revisit one day.

1. [Understanding Dropout](https://papers.nips.cc/paper/2013/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf)
2. [Dropout Training as Adaptive Regularization](https://proceedings.neurips.cc/paper/2013/file/38db3aed920cf82ab059bfccbd02be6a-Paper.pdf)
3. [Fast dropout training](https://nlp.stanford.edu/pubs/sidaw13fast.pdf)
4. [Stochastic Gradient Descent as Approximate Bayesian Inference](https://arxiv.org/pdf/1704.04289.pdf)
5. [Analysis of dropout learning regarded as ensemble learning](https://arxiv.org/pdf/1706.06859.pdf)
6. [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf)


# Lexical Analysis Statistic Summary

This is just a little cheat sheet for some stats from the paper Using Statistics in Lexical Analysis by Church, Gale, Hanks, and Hindle.

## Mutual Information

This is based on two events $x, y$ and is simply given by

\[[I(x;y) = \log_2 \bigg( \frac{P(x,y)}{P(x) P(y)}\bigg)\]]

So if $x,y$ are independent, this quantity is zero. However, it's difficult to get values that are very negative. The reason for this is that if x and y are fairly rare words (say one in ten thousand), then we would expect, by chance, to find the pair in every hundred million pairs. This is a rare event, and to confidently conclude the actual occurence is smaller still would require an extremely large corpus. This is a general problem of estimating rare events. 

## t-tests

If we want to say something like "powerful tea" is less common than "strong tea" using mutual information, we will run in to the problem of estimating a rare event's probability. A t-test allows us to get good results, however. Here is how the t-test works: We can try to estimate $P(z\|x)$ and $P(z\|y)$ by counting occurences $f(x,z)$ and $f(y,z)$ of the pairs and dividing by $f(x)$ and $f(y)$. The t-statistic

\[[\frac{P(z\|x) - P(z\|y)}{\sqrt{\sigma^2[P(z\|x)] + \sigma^2[P(z\|y)]}}\]]

is then a good way to attempt to reject a null hypothesis that the two quantities $P(z\|x)$ and $P(z\|y)$ are equal, in addition to giving a way to rank differences. As already mentioned, $P(z\|x)$ can be estimated by a ratio of counts. $\sigma^2[P(z\|x)]$ is the variance of this estimator, which is something like $\frac{p(1-p)}{n}$, where $p$ is the true probability and $n$ is the number of samples. For our purposes, we can approximate $1-p \approx 1$ and estimate it as $\frac{f(x,z)}{f(x)^2}$. 



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






