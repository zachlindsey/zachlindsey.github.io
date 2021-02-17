---
title: Math & Code Blog
layout: default
---

# The Satisfiability Threshold Conjecture

I recently learned about this conjecture that still remains open, despite being a fairly basic observation. Let's build up to it. First, we need to understand the SAT problem. Suppose we have a list of axioms like

* it cannot rain and snow
* if it is cold, it is not hot
* if it is hot, it is not cold
* if there is snow, it is cold
* it is hot

We might ask ourselves the question: Is there some weather pattern that satisfies these rules? It has to be hot, and so cannot be cold. And if it's neither raining nor snowing, all the other conditions are satisfied. You can check "hot, not cold, not raining, not snowing" makes all five conditions check out. This is the basic idea behind a SAT problem - we have a list of axioms like the ones above, using only simple logical operators "and", "or", "not", as well as some basic properties like "it is snowing", as we want to know if there is some assignment of true or false to each of the properties that makes all the axioms true.

## CNFs

What we actually work with is sentence in propositiona logic. This has as its basic building blocks the *atomic sentences*, which can be symbols like $P, Q, R, \ldots$ that stand in for statments like "it is raining", as well as more complicated sentences build up from the atomic ones using $\vee, \wedge, \neg$, the mathematical symbols for "or", "and", and "not".