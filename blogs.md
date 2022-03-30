---
title: Math & Code Blog
layout: default
use_math: true
---

# Old Papers Project - ResNet

Title: Deep Residual Learning for Image Recognition
Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
(WIP)

This month has been busy, so I don't have a big pile of experiments to talk about! This is partially because this paper's dataset is pretty beefy compared to the other ones talked about so far, since it's 155G compressed! So to chew on it, I decided to get some disks and compute instances on Google Cloud spun up. That all takes time!

Anyway, this paper introduces the ResNet architecture. The "Res" stands for "residual", which is the main idea put forward in the paper. The authors noted that very deep nets start to perform *worse* than shallower versions. They suspect that is is not because of vanishing gradient issues, since batch normalization should fix it. Instead, their architecture attacks the representation learned by the nets from layer to layer.

Recall in deep vision nets, each layer is trying to learn some function 

\[[\mathcal{F}(X; W,b) = \sigma(WX + b)\]]

Note that if we set $W = I$ to be the identity matrix and $b = 0$, this function is the identity map before the activation. Furthermore, if we have two of these functions and the second has $W=I, b=0$, then

\[[\mathcal{F}(\mathcal{F}(X;W,b);I,0) = \mathcal{F}(X;W,b)\]]

so that the second layer does nothing to the output. So how can it be that a deeper network can reproduce the output of the shallower network, but does not?

A place to start looking is to remember how the networks start the process. Typically the components of $W$ are normally distributed around 0 and $b = 0$ at the start, so the layers $\mathcal{F}$ of the network start quite far from the "do nothing" setting. 

The solution this paper introduces is to introduce skip connections. For instance, if $L_1, L_2$ are two layers of the network, instead of passing $L_2(L_1(X))$ into the next layer in the network, we pass $X + L_2(L_1(X))$. This means 

1. The network is only learning the "residual" between the input X and the output, hence the name of the architecture.
2. Since $L_1, L_2$ will start out close to the zero map, $X + L_2(L_1(X))$ will start near the identity, giving the deeper network a better starting point.



# Old Papers Project - Adam Optimizer
(WIP)

Title: ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION
Authors: Diederik P. Kingma, Jimmy Lei Ba

The Adam optimizer is ubiquitous in ML, running the optimization for many projects. So let's dig into it a little bit! The main issue is whether or not boring stochastic gradient descent, 


\[[\Theta_{t+1} = \Theta_t - \alpha \nabla f(\Theta_t) \]]

can be improved upon. Here, $\Theta$ is the weights of a neural net (or any other parameters over which we're optimizing), $f$ is some loss function, and $\alpha$ is a step size.

Some ideas that inspired the Adam optimizer:
1. Keep track of "momentum". That is, instead of just stepping in the direction of the current gradient, also make use of past gradients in some way.
2. Scale the step size for each update somehow. (AdaGrad)
3. Scale the gradient to be roughly the same size for each minibatch. (RMSProp)

We will see that Adam blends all three of these together. Here is the algorithm, written in pseudocode:

```
def adam(
    f, // sequence of functions to optimize
    alpha, // step size
    beta1, // first moment decay
    beta2, // second moment decay
    theta, // some initial config,
    eps // some small number
    ):
    m = 0 // first moment
    v = 0 // second moment
    t = 0 // timestep
    converged = False
    while not converged:
        t += 1
        grad = f.next().grad(theta)

        m = beta1*m + (1-beta1)*grad // update first moment
        v = beta2*v + (1-beta2)*grad**2 // update second moment
        // note - **2 here is just pointwise

        mhat = m/(1-beta1**t) // debias
        vhat = v/(1-beta2**t) // debias

        theta -= alpha * mhat / (sqrt(vhat) + eps)

        converged = check_convergence()
```

## Logistic Regression

The first task the paper discusses is logistic regression on the MINST dataset. So something really simple like 
```
class MNistLogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(28*28, 10)
        
    def forward(self, X):
        X = torch.flatten(X,start_dim=1)
        X = self.layer(X)
        return X
```
seems to be what they have in mind, with the cross entropy loss. For evaluation, they compare their Adam optimizer with SGD and Adagrad. Thankfully for me, all of these are implemented in Pytorch! Unsurprisingly, the paper does not give me any details about hyperparameters other than "they were obtained by a dense grid search". Hmmm, ok.

That sounded tedious and not fun, so I decided to try and use Optuna to do the hyperparameter search. 

```
import optuna

def objective(trial):
    lr = trial.suggest_float('lr',0,1)
#     lr_decay = trial.suggest_float('lr_decay', 0,1)
    beta1 = trial.suggest_float('beta1', 0, 1)
    beta2 = trial.suggest_float('beta2', 0, 1)
#     momentum = trial.suggest_float('momentum', 0, 1)
#     weight_decay = trial.suggest_float('weight_decay', 0,1)
    l2_reg = trial.suggest_float('l2', 0, 1)
    
    lr = math.exp(-1/fancy_lr)
    
    model =  MNistLogReg()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1,beta2), weight_decay = l2_reg)
#     optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay, nesterov = True)
#     optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
    
    val_errors = train_MNIST_logreg(model, optimizer, criterion, verbose = False)
    return min(val_errors)

study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

All was well until it was time to compare them. First off, it seems the losses all turned out *way* too high and the paper's findings were definitely not held up! The loss of the best optimizer, Adagrad, looked like 

```
Epoch 12/99
----------
100%|██████████| 7500/7500 [00:20<00:00, 365.07it/s]
train Loss: 23.9768
train Error: 0.0895
100%|██████████| 1250/1250 [00:03<00:00, 376.82it/s]
val Loss: 28.5779
val Error: 0.0984
```

Assuming the paper's "training cost" is the log-loss, this means that this model is not doing well! Here were the percentage errors on the validation dataset for each model.

<center>
    <img src='/images/optimizer_duel_MNIST.png'>
    <figcaption>blue - adam; orange - sgd; green - adagrad</figcaption>
</center>

Adam did the worst! What gives? Two possibilities:
1. The optuna hyperparameter ranges are not close to optimal. If the best lr is 1e-4, my sampling method will probably not find it.
2. The paper mentions scaling learning rates by a `sqrt(t)` term. I ignored this, but perhaps it is the secret sauce for Adam.

## Ha ha, I am an idiot

So the real answer is a little bit of (1) above, but mostly... *I forgot to normalize the data*. After fixing this issue and using the nifty `log=True` flag in Optuna's `suggest_float` for the learning rate, I was able to obtain log losses in the range reported by the paper. However, that did not change the fact that I wasn't really able to reproduce the superiority of Adam over SGD and Adagrad. Here are the new and improved classification errors for the validation set:

<center>
    <img src='/images/optimizer_duel_MNIST2.png'>
    <figcaption>blue - adam; orange - sgd; green - adagrad</figcaption>
</center>

So Adam does seem to have an edge on Adagrad here, but both lose out to plain old SGD! Maybe Optuna's randomness gave SGD a lucky break.

## Fun Fact

So the paper has a "proof" of convergence. If you carefully check out this proof, it doesn't take too long to find some fishy statements. For instance, check out this lemma:

<center>
    <img src = '/images/adam_lemma_10_3.png'>
</center>

The `g_{1:t}` term is just the vector defined by `g_{1:t}[i] = g_t[i]`. I think this lemma is *usually* true, but it's not hard to find sequences of `g` that violate it. For example, try this:

```
import math


Ginf = 0
LHS = 0
square_sum = 0

T = 2
for t in range(1,T+1):
    gt = 0.1
    LHS += math.sqrt(gt**2/t)
    square_sum += gt**2
    Ginf = max(Ginf, abs(gt))
RHS = 2*Ginf*math.sqrt(square_sum)

print('LHS=',LHS)
print('RHS=',RHS)

LHS= 0.17071067811865476
RHS= 0.02828427124746191
```

Hmm...


## More Resources

1. https://openreview.net/pdf?id=ryQu7f-RZ
2. http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
3. https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf (adagrad - I think)

# TESTING TESTING TESTING
(WIP)

Just a list of testing best practices pulled from [here](https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices) and shortened/summarized.

<center><img src='/images/patrick_testing.png'></center>

## Good Names

Give the tests descriptive names that indicate what is being tested, the context of the test, and the behavior. You should be writing good names for *everything*, so this should not be a surprising rule!

## AAA

These stand for Arrange, Act, Assert. Each of these steps should happen separately. For instance, this is bad.

```
class TestDogClass(unittest.TestCase):
    def test_dog_barks_loudly(self):
        
        dog = Dog()

        self.assertEquals(dog.bark(), "WOOF!!!!")
```

Note that the *arrange* happens on one line, but the *act* and *assert* happens together. Don't do it!

```
class TestDogClass(unittest.TestCase):
    def test_dog_barks_loudly(self):
        
        dog = Dog()

        the_bark = dog.bark()

        self.assertEquals(the_bark, "WOOF!!!!")
```

## Lazy Tests

A test should be mininmally passing. So don't non-default or non-zero values to models being constructed for the test. It's distracting and might lead to errors.

```
class TestDogClass(unittest.TestCase):
    def test_dog_barks_loudly(self):
        
        dog = Dog(breed = 'chihuahua', weight_in_lbs = 4.6) # BAD - why set these?

        the_bark = dog.bark()

        self.assertEquals(the_bark, "WOOF!!!!")
```
## Ambiguous Strings

If a test includes a string that is somehow meaningful, but that meaning is not conveyed in the test or the name of the string, it leads to confusion and possible bugs.

```
class TestDogClass(unittest.TestCase):
    def test_dog_init(self):
        self.assertException(Exception, Dog, name="Peyton") 
        # BAD - what is the significance of this name?
```

```
class TestDogClass(unittest.TestCase):
    def test_dog_init_my_dog_is_reserved(self): # BETTER - the name is more descriptive!

        reserved_dog_name = "Peyton" 
        # BETTER - now the name gives the reader some idea of what the string means

        self.assertException(Exception, Dog, name=reserved_dog_name) 
```


## Avoid Logic

Tests should be simple. Overly complicated logic in the tests is *not* simple, so if you find your test is full of `if`, `while`, `for`, and `switch` statements, perhaps it should be split into more tests.

```
class TestDogClass(unittest.TestCase):
    def test_trick(self):
        tricks = ['jump', 'shake', 'speak']
        dog = Dog()
        for trick in tricks:
            result = dog.do_trick(trick);
            self.assertEqual(result, true)
```

The above test makes multiple assertions and has a for loop. Why not just break each of those tricks into its own test?

```
class TestDogClass(unittest.TestCase):
    def test_dog_gets_full(self):
        for num_treats in range(20):
            if num_treats < 5:
                eats_the_treats = dog.eat_treats(num_treats)
                self.assertEqual(eats_the_treats, true)
            else:
                eats_the_treats = dog.eat_treats(num_treats)
                self.assertEqual(eats_the_treats, false)
```

This is an especially silly test! Why not just have the dog eat 4 treats and 5 treats in two separate tests? It does too much, and the logic is a code smell that clues you in to this fact.

## Setup and Teardown Bring Heartache
Perhaps this is controversial, because I personally like using setup and teardown methods! In `unittest`, you can give one of your test classes a `setUp` method to run before each test.

```
class TestDogClass(unittest.TestCase):
    def setUp(self):
        self.dog = Dog()

    def test_dog_barks(self):
        the_bark = self.dog.speak()

        self.assertEqual(the_bark, 'Woof!')
```

What's wrong with this? It keeps the code DRY, right? Well, yes! But keep in mind that it forces *all tests* to setup with the *same* code. What if a test needs a different set up? Your only option is to create another `TestCase` class! However, you could instead simply ignore the `setUp` functionality of your test suite in favor of constructors, like this:

```
class TestDogClass(unittest.TestCase):
    def create_default_dog(self):
        self.dog = Dog()


    def test_dog_barks(self):
        create_default_dog()

        the_bark = self.dog.speak()

        self.assertEqual(the_bark, 'Woof!')
```

This does two things for us. First, it makes the `arrange` step in the code clear. It also gives us more flexibility for `Dog` tests that do not use the default dog.


## Lazy Tests, Take 2

## Test Public Methods

## Stub stub stub



# Old Papers Project - Dropout

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

This also did not work...

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

![street view no drop lr decay](/images/streetview_nodrop_lrdecay.png)

Success!

## Feature Sparsity

[Kaggle Notebook](https://www.kaggle.com/zlindsey/mnist-dropout-feature-examination)

One last part of the paper that I succesfully reproduced concerned how dropout affects the sparsity of features. The idea is that without dropout, the weights of the different units can become codependent in odd ways, relying on each other for corrections. For instance, you can express `10` as `10 = 59 + 299 - 348`, and a neural network might learn to do just that if no regularization is put into place, even if it seems far more natural to just express `10 = 10 + 0 + 0`.

So we set up two autoencoders. Each takes the `28 x 28` MNIST images as input, feeds this into a layer of 256 units, and then has a `28 x 28` output. Mean squared error is used on the original input to try to get the neural nets to learn to reconstruct the input. In one, we put a dropout of `p = 0.5` on the middle layer, and the other has no dropout.

<center><img src='/images/weights_nodropout.png'><figcaption>Weights of autoencoder with no dropout.</figcaption></center>

<center><img src='/images/reconstruction_nodropout.png'><figcaption>Example of input to of autoencoder with no dropout.</figcaption></center>

<center><img src='/images/reconstruction_output_nodropout.png'><figcaption>Output for above inputs for autoencoder with no dropout.</figcaption></center>

```
Epoch 99/99
----------
100%|██████████| 15000/15000 [00:56<00:00, 264.10it/s]
train Loss: 26.0594
100%|██████████| 2500/2500 [00:06<00:00, 371.86it/s]
val Loss: 25.9332
```

The reconstructions a pretty good! But look at the first image. These are the weights of each of the 256 internal units. That is, this is what each of the hidden units are "looking for" in the image in order to activate. Notice how they're all quite blurry and random looking, and it's not at all clear how a digit could be built from them.

On the other hand, here are the same set of images for the network trained with dropout.

<center><img src='/images/weights_dropout.png'><figcaption>Weights of autoencoder with 0.5 dropout.</figcaption></center>

<center><img src='/images/reconstruction_dropout.png'><figcaption>Example of input to of autoencoder with 0.5 dropout.</figcaption></center>

<center><img src='/images/reconstruction_output_dropout.png'><figcaption>Output for above inputs for autoencoder with no dropout.</figcaption></center>

```
Epoch 99/99
----------
100%|██████████| 15000/15000 [00:55<00:00, 268.41it/s]
train Loss: 87.2140
100%|██████████| 2500/2500 [00:06<00:00, 379.09it/s]
val Loss: 64.3704
```

The reconstructions still look pretty good, even if the loss is much higher, but the real stunning difference here is what the weights look like! Notice that instead of random, noisy-looking blurs, each neuron has learned to focus on making a single stroke or dot! This is evidently because the output layer cannot so delicately reconstruct the result, so needs more straightforward hidden units to function.

## Conclusion

I would have liked to check more results! The original paper investigated some text and audio datasets, too. It looks like, with some tinkering, dropout can help give a network that's overfitting an extra edge as well as regularize the weights. However, doing it is not a simple matter of throwing it in! The network with dropout requires care to find proper hyperparameters, lr decay, etc. Some things to remember...

+ Normalize the data! The feature sparsity notebook did not work until I realized that I needed to do this.
+ Use LR decay. Some networks get stuck after training at a certain LR for a period of time, and need the decrease to converge. I guess this is especially important for dropout, since extra noise is injected.

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






