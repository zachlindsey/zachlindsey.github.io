# Intro

The Fourier transform and inverse transform are formally defined as something like hi

$$
\begin{align*}
\mathcal{F}\[f\](\omega) &= \frac{1}{\sqrt{2\pi}} \int f(t) e^{-\omega i t} dt \\\\
\mathcal{F}^{-1}\[\hat{f}](t) &= \frac{1}{\sqrt{2\pi}} \int \hat{f}(\omega) e^{\omega i t} d\omega
\end{align*}
$$


I'll sometimes write $\hat{f}$ for $\mathcal{F}[f]$ or $\check{f}$ for $\mathcal{F}^{-1}[f]$.  If we want to live in a fantasy world, we pretend
* Both the transform and inverse transform always exist
* The inverse transform is the inverse of the transform
* The integral might actually be a sum!

But this is sadly not the case. Instead, there are a variety of different "spaces of things" on which the transforms are defined, and it's often not so clear which of these nice properties hold. And then there are things like Fourier series and discrete Fourier transforms, which complicate things even more. So let's untangle all this mess! Fourier analysis has a funny way of making you either pay  with  complications somewhere - either the transforms and their properties need care or the objects on which you apply them need care. This means I have to choose an end of the spectrum (no pun intended) to start on, and so I'll choose a somewhat complicated space of functions with nice transform properties. Here we go!

# Schwartz Space

## Definition of $\mathcal{S}$

Suppose that $f:\mathbb{R} \rightarrow \mathbb{C}$ is a smooth (infinitely differentiable) function. For $n,m$ nonnegative integers, we can define a norm $$\\|f \\|_\{n,m\} = \sup\_{x \in \mathbb{R}} |x^n f^{(m)}(x)| $$

and from this define the *Schwartz space* $\mathcal{S}$ of all such functions for which all of these norms are finite:

$$ \mathcal{S} := \\{ f \in C^{\infty}(\mathbb{R}) : \forall_n \forall_m \\|f \\|_\{n,m\} < \infty \\} $$

These are very nice functions - smooth, with strong decay properties - that allow a nice development of Fourier analysis. Some facts that are easy to see from the definition:

* $\mathcal{S}$ is closed under differentiation and multiplication by polynomials
* If $f,g \in \mathcal{S}$, so is $fg$.

## $\mathcal{F}$ on $\mathcal{S}$

**Fact 1** If $f \in \mathcal{S}$, then both $\mathcal{F}\[f\]$ and $\mathcal{F}^{-1}\[f\]$ exist and belong to $\mathcal{S}$.

**Proof** I'll just stick to showing everything for $\hat{f} = \mathcal{F}[f]$. The inverse follows almost exactly the same way. 

Since $|x^2 f(x)| \leq C$ is finite, we get $$\int |f(x) e^{- i \omega x}| dt \leq \int_{-1}^1 |f(x)| dx + \int_{[-1,1]^c} \frac{C}{|x|^2} dx$$ converges absolutely, so $\hat{f}$ exists. A similar bound will arise for other integrals involving $x^n f^{(m)}{x}$ that will be bounded in the same way.

We can differentiate under the integral sign, as well: $- i x f(x) e^{- i \omega x}$ is the partial derivative of the integrand wrt $\omega$, which is bounded by $\frac{1}{|x|^2}$ at the tails and smooth near 0, and so $$\frac{d}{d\omega} \hat{f}(\omega) = \frac{1}{\sqrt{2\pi}} \int -i x f(x) e^{- i \omega x} dx = -i \widehat{xf}(\omega)$$

By induction, we have the well-known identity that says that "the Fourier transform turns multiplication by $x$ into differentiation":

$$ \frac{d^n}{d\omega^n} \hat{f}= (-i)^n \widehat{x^nf}$$

In particular, $\hat{f} \in C^\infty$. On the other hand, we can also consider $\widehat{f'}$. Integrating by parts, we get

$$\widehat{f'}(\omega) = \frac{1}{\sqrt{2\pi}}\int f'(x) e^{-ix\omega} dx = \frac{i\omega}{\sqrt{2\pi}} \int f(x) e^{-ix\omega}dx = i\omega \hat{f}(\omega)$$ and so $$\widehat{f^{(n)}}(\omega) = (i\omega)^n \hat{f}(\omega).$$


In other words "the Fourier transform turns differentiation into multiplication by $\omega$". These identities make estimating the Schwarz norms more straightforward:

$$
\begin{align*}
&\bigg| \omega^n \frac{d^m}{d\omega^m} \hat{f} \bigg| \\\\
= &\bigg| (-i)^{m-n} \widehat{  \frac{d^n}{dx^n} x^m f } \bigg| \\\\
\leq &\frac{1}{\sqrt{2\pi}}\int \bigg|\frac{d^n}{dx^n} x^m f (x) e^{-i\omega x}\bigg| dx  \\\\
\leq &\frac{1}{\sqrt{2\pi}}\int \bigg|\frac{d^n}{dx^n} x^m f (x) \bigg| dx 
\end{align*}
$$

Now, this last integrand is in $\mathcal{S}$ so is the whole integral bounded. ☐

## Inversion

**Fact 2** $\mathcal{F}^{\pm 1}$ are inverses on $\mathcal{S}$.

**Proof** Let's start with $\mathcal{F}^{-1} \mathcal{F}$. We need to compute

$$ \frac{1}{2\pi} \int \int f(t) e^{-i\omega t} e^{i \omega x} dt d\omega $$

We want to apply Fubini and compute $$\int e^{i\omega(x-t)} d\omega,$$ but this is actually a funny integral. Use finite bounds and assume $x \neq t$ and you get $$\int_{-L}^L e^{i\omega(x-t)} d\omega = \frac{e^{i\omega (x-t)}}{i(x-t)} \bigg|_{-L}^L = \frac{e^{iL(x-t)} - e^{-iL(x-t)}}{i(x-t)} = \frac{2\sin(L(x-t))}{x-t}$$ When $x = t$, the integral is just $2L$. In either case, it's clear that taking $L \rightarrow \infty$ is going to cause some problems, which is some foreshadowing of tricks to come. However, we can be sneaky. When $L = \frac{2\pi k}{x-t}$, this integral is just 0. So we're going to build some regions to take advantage of this. Here's the plan. (TODO - A picture...) Let $B_n(t)$ be defined as

$$B_n(t) = \begin{cases} \frac{2\pi n}{t-x} &\text{if } t > x+ \frac{1}{n} \\\\  \frac{2\pi n}{x-t} &\text{if } t < x - \frac{1}{n} \\\\ 2\pi n^2 &\text{else} \end{cases}$$

Then let $R_n$ be the region enclosed by $\pm B_n$. Let's integrate over $R_n$:

$$
\begin{align*}
&\int\int_{R_n}  f(t) e^{i\omega (x-t)} dt d\omega \\\\
= &\int_{|t-x| \geq \frac{1}{n}} f(t) \int_{-B_n(t)}^{B_n(t)} e^{i\omega(x-t)} d\omega dt + \int_{x-\frac{1}{n}}^{x+\frac{1}{n}} \int_{-2\pi n^2}^{2\pi n^2} f(t) e^{i\omega(x-t)} d\omega dt \\\\
=\ &0 + \int_{x-\frac{1}{n}}^{x+\frac{1}{n}} f(t) \frac{2 \sin(2\pi n^2 (x-t))}{x-t} dt \\\\
=\ &2 \int_{-\frac{1}{n}}^{\frac{1}{n}} f(u+x) \frac{\sin(2\pi n^2 u)}{u} du \\\\
=\ &2 \int_{-2\pi n}^{2\pi n} f\bigg(x + \frac{z}{2\pi n^2}\bigg) \frac{\sin(z)}{z} dz
\end{align*}
$$

Because of $f$'s decay properties, we can take the limit as $n \rightarrow \infty$ and get

$$2 f(x) \int \frac{\sin(z)}{z} dz = 2\pi f(x)$$

And wouldn't you know... that's exactly what we needed! The $\mathcal{F} \mathcal{F}^{-1}$ is almost the same, since it wants to compute $$ \frac{1}{2\pi} \int \int f(t) e^{i\omega t} e^{-i \omega x} dt d\omega $$ which differs only by the sign of the exponential. ☐

## Examples 

Coming soon! The Gaussian will be the mascot. TODO

## Summary

This is nice work so far! We have found a "base camp" of very nice functions on which the Fourier transform has everything we could possibly want:
+ Always defined
+ Inverse transform is always defined
+ Forward and inverse transform cancel
+ Usual nice properties, like multiplication by $x$ => differentiation and differentiation => multplication by $\omega$.

The issue is, of course, that the "base camp" is lacking in another way - there are lots of functions we want to do Fourier analysis on that definitely do not lie in $\mathcal{S}$! Periodic functions are a notable absence. And what do we make of functions only defined on $\mathbb{Z}$? Let's continue climbing...

# Gathering Supplies...

I initially wanted to just introduce things "as needed", but this doesn't actually help exposition, as it turns out! So now, I want to gather up a few useful tools that we'll need to move from $\mathcal{S}$ to more interesting, flexible spaces of functions.

## $L^p$ spaces and more!

What kinds of spaces of functions will we be interested in? Well, there's the obvious $L^p$ spaces - measurable functions $f:\mathbb{R} \rightarrow \mathbb{C}$ for which $$\\|f\\|_p := \bigg(\int |f|^p\bigg)^\frac{1}{p} < \infty,$$ with the caveat that we think of two functions as being "the same" if $f \neq g$ has **measure zero**. This isn't the place to get into the details of what measurable or measure zero means, but rest assured almost any function you can think of is measurable and "measure zero" essentially means "length zero". The norm $\\| f \\|_p$ has lots of nice properties like the triangle inequality. $L^p$ is also complete, meaning that if I have a sequence of functions $f_n$ such that $\\| f_n - f_m\\|_p$ gets really small, then there is a function function $f$ with $f_n \rightarrow f$. One detail: when we say $f_n \rightarrow f$, we sometimes need to be more careful. When we're working in $L^p$, this means convergence in $L^p$: $$f\_n \rightarrow f \text{ means }\\|f\_n -f \\|\_p \rightarrow 0,$$ and I'll write $f\_n \stackrel{p}{\rightarrow} f$ if this needs to be made clear.

We'll actually find it conventient to cook up more interesting function spaces. For instance, what if we want $\int |x f(x)| dx < \infty$? Or $\int |f'(x)| dx < \infty$? Here's what we'll use to handle all of this nicely: Fix weighting functions $w\_1, \ldots, w\_n$, derivative orders $k\_1,\ldots, k\_n$, and norms $p\_1, \ldots, p\_n$. Then if $f: \mathbb{R} \rightarrow \mathbb{C}$ is measurable, has derivatives up to order $\max(k\_i)$ a.e., and $f^{(k)} = \int f^{(k+1)}$ a.e., then we can put norms on it like $\\| w_i f^{(k_i)} \\|_{p_i}$, and consider such functions for which $$ \sum\_i \\| w\_i f^{(k\_i)} \\|\_{p\_i} < \infty$$

I suppose we can denote this monstrosity of a space by $L^{p_\bullet; w_\bullet; k_\bullet}$ or something. Note these spaces are no longer complete, in general:
* If $w_i = 0$ on a set of positive measure, then functions can converge while having very different values on that set
* If $k_i > 0$ and derivatives get involved, you can obviously have functions with the same derivative that are not the same. :)
* Even if you throw in a $w_i = 1; k_i = 0$ term to force $L^{p_i}$ convergence to *something*, there's no guarantee it will have the derivatives you want - I leave cooking up an example as an "exercise".

**Holder's Inequality** An important relationship between $L^p$ spaces is this: If $1 \leq p,q \leq \infty$ and $f \in L^p, g \in L^q$, then $fg \in L^1$ provided $\frac{1}{p} + \frac{1}{q} = 1$. In fact, $$\\| fg\\|\_1 \leq \\|f\\|\_p \\|g\\|\_q.$$


### Density Arguments in $L^p$

Proving things about general elements of $L^p$ is usually hard. But there is a trick to let you prove things about them by reasoning about very simple functions. How simple? How about this: $\mathbb{I}\_{[a,b)}$, the indicator for an interval with $b-a < \infty$? That's very simple! These functions are all in $L^p$, since $\mathbb{I}\_\{[a,b)\}^p = \mathbb{I}\_\{[a,b)\}$. We'll then define a **simple function** as a finite sum $$s(x) = \sum\_n c\_n \mathbb{I}_\{[a_n,b_n)\}(x),$$ where each $b_n - a_n < \infty$. The amazing fact about these functions is...

> **Density** The space of simple functions is dense in $L^p$ for all $1 \leq p < \infty$. That is, for any $f \in L^p$, there is a sequence $s_n$ of simple functions with $\\|s_n - f\\|_p \rightarrow 0$.

This lets you prove some things for simple functions and conclude it for all $L^p$ "for free"! 

*Proof*  
Begin by writing $f = f_+ - f_-$, where $f_+ = \max(f,0)$ and $f_- = \max(-f,0)$. This lets us assume that $f \geq 0$, since if $s_n \rightarrow f_+$ and $t_n \rightarrow f_-$, then $s_n - t_n \rightarrow f_+ - f_- = f$. 

Next, we can assume that $f$ has compact support. Let $f_n = f \mathbb{I}_{[-n,n]}$. Then $$\\| f - f\_n \\|^p\_p = \int \mathbb{I}\_{|x| > n}|f(x)|^p dx.$$ This integral is bounded by the integrable function $|f|^p$, so we can apply the DCT to see it goes to zero. So if we can approximate $f_n$ by simple functions in $L^p$, we just pick $s_n$ with $\\| f_n - s_n \\|\_p < \frac{1}{n},$ and then $f\_n \rightarrow f$ implies $s\_n \rightarrow f$, too.

Next, we do "quantization". Consider the intervals $[\frac{k}{n}, \frac{k+1}{n})$ for $k,n \in \mathbb{N}$ and the corresponding sets $A_{n,k} := \\{x : f(x) \in [\frac{k}{n}, \frac{k+1}{n}) \\}$. For fixed $n$, these sets are disjoint, and the function $$q_n(x) = \sum_k \frac{k}{n} \mathbb{I}\_{A\_{n,k}}$$ satisfies
+ $0 \leq q_n$
+ $f - \frac{1}{n} \leq  q_n \leq f$
+ $q_n$ takes on the "quantized values" $\\{0, \frac{1}{n}, \frac{2}{n}, \ldots \\}$

We can compute

$$\begin{align*}
&\\| f - q_n \\|^p_p \\\\
= &\int |f(x) - q_n(x)|^p dx \\\\
= &\sum_k \int_{A\_{n,k}} |f(x) - q_n(x) |^p dx \\\\
\leq &\sum_k \frac{1}{n^p} |A_{n,k}| \\\\
= &\frac{1}{n^p} | \text{supp}(f) |
\end{align*}$$

Where $\text{supp}(f)$ is the (bounded) support of $f$. Hence $q_n \rightarrow f$ in $L^p$.

A small remark - this fails for $L^\infty$! The reason should be clear - the maximum distance between a simple function and the constant one is always 1!

**Finite Sums**
Now we show that the infinite sum can be truncated. Recall that we had $f = \sum_k a_k \mathbb{I}\_{A_k}$ with $\cup_k A_k$ bounded and all $A_k$ disjoint. Since $f \in L^p$, we know that the series $\\| f \\|^p_p = \sum_k |A_k| a_k^p < \infty$. The tails $\sum_{k > n} |A_k| a_k^p$ of that series must converge to zero, but if $f_n = \sum_{k=0}^n a_k \mathbb{I}\_{A_k}$, then $f - f_n = 0$ on $\cup_{k=1}^n A_k$, while $f - f_n = f$ on $\cup_{k=n+1}^\infty A_k$ so 

$$\\| f - f_n \\|^p_p = \int | f - f_n |^p = \sum_k \int_{A_k} |f - f_n|^p = \sum_{k \geq n+1} |f|^p = \sum_{k > n} |A_k| a_k^p \rightarrow 0$$

**Intervals**
And I stop here! We will just use this fact and call it a day: If $A$ is a Lesbesgue measurable set and $\varepsilon > 0$, there is a finite union of intervals so that $|A \triangle \cup_i I_i| < \varepsilon$.

### Application - Continuity of Translation on $L^p$.

Let's immediately use this technique in practice to show that the operator $\tau\_h : L^p \rightarrow L^p$ defined as $(\tau\_h f)(x) = f(x+h)$ is continuous. That is, $$\lim\_{h \rightarrow 0} \\| \tau\_h f - f \\|\_p = 0.$$ If you think this is simple without using any fancy techniques, try it! But we will use the density argument. If $f$ is a simple function $\sum c\_n \mathbb{I}\_{(a_n, b_n)}$, then $$\int |\tau\_h f - f|^p \leq \sum |c\_n|^p | (a_n+h, b_n+h) \triangle (a_n, b_n)| \leq \sum |2h| |c\_n|^p  $$ Let $h \rightarrow 0$ and we're done. If $f$ is an arbitrary $L^p$ function and $\varepsilon > 0$, then pick a simple interval function with $\\| f - f_n \\|_p < \varepsilon$. Then $\\| \tau_h f - \tau_h f_n \\|_p < \varepsilon$, as well. Hence

$$\begin{align*}
\\| f - \tau_h f\\|_p &\leq \\| f - f_n \\|_p + \\| f_n - \tau_h f_n \\|_p + \\| \tau_h f_n - \tau_h f\\|_p \\\\
&< 2\varepsilon + \\| f_n - \tau_h f_n \\|_p
\end{align*}$$

Now let $h \rightarrow 0$. Note we used the fact that $\tau_h$ is an isometry on $L^p$, so $\\| \tau_h f \\|_p = \\| f \\|_p$.

**Generalization** Suppose $T: L^p \rightarrow L^q$ is any linear map with a constant $|T|$ such that $$\\| Tf \\|\_q \leq |T| \\| f\\|\_p$$ when $f$ is a simple function. Then $$\\| T f \\|\_q \leq |T| \\| f\\|\_p$$ for all functions $f \in L^p$.

## Convolutions, Approximate Identities
Life will throw many nasty functions at us, and we must find a way to deal with them. Much like applying an Instragram filter can make someone look much nicer, math has provided us with a "functional Instagram filter" in the form of convolutions. The definition:

$$ (f \ast g)(x) = \int f(x-t) g(t) dt $$

when it exists, of course. Some basic properties when everything is defined:
+ $f \ast g = g \ast f$
+ $\ast$ is linear in each input 

For existence, things are pretty straightforward when both functions are $L^1$, since 

$$\int |(f \ast g)(x)| dx \leq \int \int |f(x-t)g(t)|dt dx = \int \int |f(x-t)| |g(t)| dx dt \leq \\|f\\|_1 \int |g(t)| dt \leq \\|f\\|_1 \\|g\\|_1.$$

Something kind of funny about this is that you don't actually know that $f \ast g$ is always defined at each point. You might think, "but wait, what about Holder?" but the exponent arithmetic doesn't work! But it's defined at enough points to somehow be integrable! So it must at least be finite almost everywhere. It kind of makes sense - if $f$ is in $L^1$ but not $L^2$, obviously $(f \ast f)(0)$ has problems. But it's okay at most other points! The intuition is that $f \ast g$ only has problems when the shift in the convolution "lines up" the singularities of $f$ with the singularities of $g$. 


### Smoothing
I mentioned that convolution makes things look nicer somehow. Here's what I mean - suppose that $\phi$ is a continuously differentible function with $\phi, \phi' \in L^1$ and $f \in L^1$. Then $\phi \ast f$ is also differentiable, and $(\phi \ast f)' = \phi' \ast f$. This is not so hard to see - since we know $\phi' \ast f \in L^1$, we can consider

$$ \frac{(\phi \ast f)(x+h) - (\phi \ast f)(x)}{h} - (\phi' \ast f)(x) = \int \bigg(\frac{\phi(x+h-s) - \phi(x-s)}{h} - \phi'(x-s)\bigg) f(s) ds = \bigg\(\frac{\tau_h \phi - \phi}{h} - \phi'\bigg\) \ast f $$

Where $\tau_h$ is just the "shift by $h$" operator. Now, if we somehow know that the derivative worked "in $L^1$", we would be done! But is it really the case that $$\lim_{h \rightarrow 0} \frac{\tau_h \phi - \phi}{h} - \phi' = 0$$ in $L^1$? Well... yes! You write $$\frac{1}{h}(\phi(x+h) - \phi(x)) = \frac{1}{h} \int_{x}^{x+h} \phi'(t) dt = \int_0^1 \phi'(x+th) dt,$$ and then you check 
$$
\begin{align*}
&\int \bigg| \int\_0^1 \phi'(x+th) dt - \phi'(x) \bigg| dx \\\\
= & \int \bigg| \int\_0^1 \phi'(x+th) - \phi'(x)  dt\bigg| dx \\\\
\leq &\int \int\_0^1 |\phi'(x+th) - \phi'(x)| dt dx \\\\
= &\int_0^1 \int |\phi'(x+th) - \phi'(x)| dx dt
\end{align*}$$

Now, the inner integral is $\leq 2\\|\phi'\\|\_1$, so we can apply the DCT to the outer integral. But then what about $$\lim_{h \rightarrow 0} \int |\phi'(x+th) - \phi'(x)|dt?$$ Recall that we already showed that translation is continuous on $L^p$!

### Riesz-Thorin Interpolation
We showed that $\ast: L^1 \times L^1 \rightarrow L^1$. It's also easy to show that $\ast: L^1 \times L^\infty \rightarrow L^\infty$, since if $f \in L^1, g \in L^\infty$, then $$\bigg| \int f(x-t) g(t) dt \bigg| \leq \int |f(x-t)| |g(t)| dt \leq \|| g \\|\_\infty \int |f(x-t)|dt = \\|g\\|\_\infty \\|f \\|\_1.$$

This seems like a lot of nothing, but actually using some fancy cool machinery, it will let us say a lot about how convolutions (and other operators, like $\mathcal{F}$) behave on various $L^p$ spaces. To set it up, note that for a fixed $f \in L^1$, we now have two operators

$$\begin{align*}
f \ast - &: L^1 \rightarrow L^1 \\\\
f \ast - &: L^\infty \rightarrow L^\infty
\end{align*}$$

These two operators have the same formula, and so agree on $L^1 \cap L^\infty$, and so give an operator $$f \ast - : L^1 + L^\infty \rightarrow L^1 + L^\infty$$

Now, $$L^1 \cap L^\infty \subseteq L^p \subseteq L^1 + L^\infty,$$ which you can see by doing similar decompositions of $f$ for each inclusion:
+ If $f \in L^1 \cap L^\infty$, then $$\int |f|^p = \int_{|f| < 1} |f|^p + \int_{|f| \geq 1} |f|^p \leq \\|f\\|_1 + \\|f\\|_\infty^p |\\{ |f| > 1\\}\| < \infty$$
+ If $f \in L^p$, then you can write $f = f\mathbb{I}\_{|f| < 1} + f \mathbb{I}\_{|f| \geq 1}$. The first is clearly in $L^\infty$ and the second is in $L^1$ since $\int_{|f| > 1} |f| \leq \int_{|f| > 1} |f|^p < \infty$.

So we actually get a map $$ f \ast - : L^p \rightarrow L^1 + L^\infty,$$ and the real magic is that *this actually lands back in $L^p$*! The following is a fairly deep, important theorem and I'm going to skip the proof, but know it's a big deal! As you'll see from all the magic we pull out with it.

**Reisz-Thorin Interpolation Theorem** Suppose $T_0: L^{p_0} \rightarrow L^{q_0}$, $T_1: L^{p_1} \rightarrow L^{q_1}$ are two linear maps with $$\\|T\_i f \\|\_{q_i} \leq \\|T\\|\_i \\| f\\|_{p\_i},$$ and $T_0 = T_1$ on $L^{p\_0} \cap L^{p\_1}$.  For $0 < \theta < 1$, let $p\_\theta, q\_\theta$ be defined by $$\frac{1}{p\_\theta} = \frac{1-\theta}{p\_0} + \frac{\theta}{p\_1},$$ and similarly for $q\_\theta$. Then...

1. The space $L^{p_\theta}$ satisfies $L^{p_0} \cap L^{p_1} \subseteq L^{p_\theta} \subseteq L^{p_0} + L^{p_1}$ and similarly for $q_\theta$.
2. Because $T_0 = T_1$ where both are defined, they jointly form a single operator $T: L^{p_0} + L^{p_1} \rightarrow L^{q_0} + L^{q_1}$.
3. (!!) The restriction $T_\theta$ of $T$ to $L^{p_\theta}$ lies in $L^{q_\theta}$
4. (!!) $\\| T_\theta \\| \leq \\| T_0 \\|^{1-\theta} \\| T_1 \\|^\theta.$


The first two results are not so surpising, but the last two are really something! They mean that you can "interpolate" bounded linear operators across $L^p$ spaces! We immediately harvest the nice result that if $f \in L^1$, then

$$f \ast -: L^p \rightarrow L^p \text{ with } \\| f \ast g \\|_p \leq \\| f \\|_1 \\|g \\|_p.$$

And we can push further. Again with $g \in L^p$, we know that

$$\begin{align*}
\- \ast g &: L^1 \rightarrow L^p, \text{ by the above,} \\\\
\- \ast g &: L^q \rightarrow L^\infty, \text{ by Holder, where } \frac{1}{p} + \frac{1}{q} = 1
\end{align*}$$

So we can apply the interpolation formula again: if we write $\frac{1}{s} = 1 - \theta  +\theta(1 - \frac{1}{p}) = 1 - \frac{\theta}{p},$ and $\frac{1}{t} = \frac{1-\theta}{p} = \frac{1}{p} - \frac{\theta}{p} = \frac{1}{p} + \frac{1}{s} - 1$, then we recover a result known as **Young's inequality**

$$\text{if } \frac{1}{p} + \frac{1}{q} = \frac{1}{r} + 1 \text{, then } \ast: L^p \times L^q \rightarrow L^r$$ and furthermore $$\\| f \ast g \\|_r \leq \\| f \\|_p \\| g \\|_q.$$

We will have a chance to make use of this later, with $\mathcal{F}$ as the target! Note that this bound means that if $\phi, \phi'$ are $L^1$ and $f \in L^p$, then we will have $(\phi \ast f)' = \phi' \ast f \in L^p$! So we really have shown that $\phi \ast -$ is a nice "smoothing filter" on $L^p$!

### Approximate Identities

We can push the filter analogy a little further. A good filter wouldn't change the original input much! So you might ask this question: Given $\varepsilon > 0$, is there a $\phi_\varepsilon \in L^1$ with $\phi_\varepsilon' \in L^1$ such that $$\\| \phi_\varepsilon \ast f - f \\|_p < \varepsilon \\| f \\|\_p$$ for all $f \in L^p$? If not, maybe just for a single $f$? Note a bit of intuition, if it were the case that there were a function $\delta \in L^1$ that satisfies $\delta \ast f = f$, then we could do this:

$$\\| \phi_\varepsilon \ast f - f \\|\_p = \\| \phi\_\varepsilon \ast f - \delta \ast f\\|\_p \\| \leq \\| \phi\_\varepsilon - \delta\\
\\|_1 \\|f \\|_p < \varepsilon,$$

and we would have an obvious path to get the bound, but this is sadly nonsense (for now). Nonetheless, the initial idea *still works*, and the (temporray) nonsense explains why we call the $\phi_\varepsilon$ **approximate identities** for $\ast$. 

To prove this, we can "cheat" by using density. Suppose that $s$ is simple with $\\|s - f\\|_p < \varepsilon$. We also need one constraint on $\phi$, namely $\int \phi =1 $. Then $$
\begin{align*}
\\| \phi\_\varepsilon \ast f - f \\|_p &\leq \\| \phi\_\varepsilon \ast f - \phi \ast s \\|_p  + \\| \phi\_\varepsilon \ast s - s \\|_p  + \\| s - f \\|_p \\\\
&\leq (\\| \phi\_\varepsilon \\|_1 + 1 ) \\| f - s\\|_p + \\| \phi\_\varepsilon \ast s - s \\|_p
\end{align*}$$

This means we just need to understand what happens for intervals! For $[a,b]$, $$
\begin{align*}
\int |\mathbb{I}\_{[a,b]} - \phi_\varepsilon \ast \mathbb{I}\_{[a,b]}|^p &= \int \bigg| \mathbb{I}\_{[a,b]}(x) - \int \mathbb{I}\_{[a,b]}(x-t) \phi_\varepsilon(t) dt \bigg|^p dx \\\\
&= \int \bigg| \mathbb{I}\_{[a,b]}(x) - \int \mathbb{I}\_{[a,b]}(x-\varepsilon s) \phi(s) ds \bigg|^p dx \\\\
&= \int \bigg| \mathbb{I}\_{[a,b]}(x) - \int\_{\frac{x-b}{\varepsilon}}^{\frac{x-a}{\varepsilon}} \phi(s) ds \bigg|^p dx \\\\
\end{align*} $$

Now that everything here is nice and bounded and compactly supported, we can apply the DCT and move $\lim_{\varepsilon \rightarrow 0}$ inside. What happens to $\lim_{\varepsilon \rightarrow 0} \int\_{\frac{x-b}{\varepsilon}}^{\frac{x-a}{\varepsilon}} \phi(s) ds?$ The upper and lower bounds diverge to $\pm \infty$ except when $x \in \{a,b\}$, and those two points are irrelevant for this. But look! If $x \notin [a,b]$, then we have the same sign for both bounds. Since $\phi$ is compactly supported, this eventually escapes the support and you get $0$. On the other hand, if $x \in (a,b)$, then the two bounds cover the entire compact support of $\phi$, giving 1! So the whole integral $\rightarrow 0$. Returning to the general case, we now know

$$\limsup_{\varepsilon \rightarrow 0} \\| \phi_\varepsilon \ast f - f \\|\_p \leq (\\| \phi\_\varepsilon \\|_1 + 1 ) \\| f - s\\|_p ,$$ and now we let $s \rightarrow f$ in $L^p$ to finish up.

To put a final cap on this little bit about smoothing, note this: If $f \in L^p$ and $\phi \in C^\infty\_c$ is smooth and compactly supported with $\int \phi =1$, then we know
* The functions $\phi_\varepsilon \ast f$ are also infinitely differentiable and remain in $L^p$
* $\lim \phi_\varepsilon \ast f \rightarrow f$ in $L^p$.
* $\phi_\varepsilon \ast f$ is compactly supported whenever $f$ is

And so from this we can conclude that the set $C^\infty_c$ is dense in $L^p$, and we have a very nice formula for constructing the approximations! Just truncate $f$'s tails and convolve!

**Key Theorem** The space $C^\infty_c$ of compactly supported smooth functions is dense in $L^p$ for all $1 \leq p \leq \infty$.

### Convolution and $\mathcal{F}$

It would be a huge omission if we didn't talk about the key formula $$\mathcal{F}[f \ast g] = \sqrt{2\pi} \mathcal{F}[f] \mathcal{F}[g]$$

that turns the fairly complicated operation of convolution into an extremely simple operation! (Ever wonder how computers can compute convolutions quickly? Hint, hint.) As is typical for things like this, we will prove it when $f, g \in C^\infty_c$ and this will let us extend by density. Compact support lets us Fubini away, so

$$
\begin{align*}
&\mathcal{F}\[f \ast g\](\omega) \\\\
= &\frac{1}{\sqrt{2\pi}}\int \int f(x-t) g(t) e^{-ix\omega} dt dx \\\\
= &\frac{1}{\sqrt{2\pi}} \int \int f(x-t) g(t) e^{-i(x-t)\omega} e^{-it\omega} dt dx \\\\
= &\frac{1}{\sqrt{2\pi}} \int \int f(s) g(t) e^{-is\omega} e^{-it\omega} ds dt \\\\
= &\sqrt{2\pi} \mathcal{F}\[f\](\omega) \mathcal{F}\[g\](\omega)
\end{align*}
$$

Note that you'll often see this identity without the extra $2\pi$ factor. This is because there's not really a perfect convention for what units to use for $\mathcal{F}$ or where to assign scaling coefficients between $\mathcal{F}$ and $\mathcal{F}^{-1}$. The result is that you either are forced to pick an annoying constant of $2\pi$ in your $\mathcal{F}[f']$ formula or this one.



# $L^2$
Now, let's move on to the next nice space for Fourier analysis: $L^2$. It  has some very nice properties: it is a Hilbert space with inner product $\langle f,g\rangle := \int f\overline{g}$.  Now, let's see about defining the Fourier transform $\int f(x) e^{-i\omega x} dx$. We don't know that $f$ is integrable alone, so we can't bound this by $\int |f(x)| dx$, so we seem a little stuck. However, we can use density of $\mathcal{S}$ in $L^2$ to help us. Before doing that, take a look at this nice interplay between $\mathcal{F}$ and the $L^2$ inner product, where for now $f, g \in \mathcal{S}$:

$$
\begin{align*}
&\langle f, \hat{g} \rangle \\\\
= & \int f(x) \overline{\hat{g}(x)} dx \\\\
= & \frac{1}{\sqrt{2\pi} }\int \int f(x)  \overline{g(y)} e^{ixy} dy dx \\\\
= & \int \hat{f}(y) g(y) dy \\\\
= & \langle \check{f}, g \rangle
\end{align*}
$$

This identity will appear later, as it gives it a number of nice properties. For now, we just want to use it for this trick:

$$\\| \hat{f} \\|_2^2 = \langle \hat{f}, \hat{f} \rangle = \langle \check{\hat{f}}, \hat{f} \rangle = \langle f, f \rangle = \\|f\\|_2^2.$$

Interesting... it seems that when we view $\mathcal{S}$ as a subspace of $L^2$, $\mathcal{F}$ is actually a linear isomorphism that perserves the norm, and so the Fourier transform and its inverse extend to all of $L^2$! The formula is this... if $f \in L^2$, to compute $\mathcal{F}[f]$, then find a sequence of Schwartz functions $f_n \in \mathcal{S}$ such that $f_n \rightarrow f$ in $L^2$, and then define $$\mathcal{F}[f] := \lim_n \mathcal{F}[f_n].$$  The same recipe follows for $\mathcal{F}^{-1}$, and pretty much any identity you can prove about $\mathcal{F}$ on $\mathcal{S}$ will extend for free. Let's revisit some of the old properties from $\mathcal{S}$ and add some new ones.

**Plancherel's Identity** The identity $$\\| \hat{f} \\\|_2 = \\|f\\|\_2$$ is fundamental. We already showed it works for functions in $\mathcal{S}$ earlier. Let's show it works for any $f \in L^2$. Following the recipe, we let $f_n \rightarrow f$ be a sequence of Schwarz functions, and so $\hat{f} = \lim \hat{f_n}$. So $$\\|f\\|_2 = \lim_n \\| f_n \\|_2 = \lim_n \\| \hat{f_n} \\|_2 = \\| \hat{f}\\|_2.$$

**$\mathcal{F}$ is continuous** Suppose that $f_n \rightarrow f$ are any sequence of functions in $L^2$ converging to some other $f \in L^2$. Is it the case that $\hat{f_n} \rightarrow \hat{f}$? Well... $$\\| \hat{f} - \hat{f_n} \\|_2 = \\| \widehat{f - f_n} \\|_2 = \\| f - f_n \\| \rightarrow 0.$$ So yes!

**$\mathcal{F} \mathcal{F}^{-1}$ is the identity** This is another good exercise in the "recipe". If $f \in L^2$, choose a sequence of approximating Schwarz functions $f_n \rightarrow f$. Then $\mathcal{F} f_n \rightarrow \mathcal{F} f$, and $f_n = \mathcal{F}^{-1} \mathcal{F} f_n \rightarrow \mathcal{F}^{-1} \mathcal{F} f$. But this means that $f_n$ converges to both $f$ and $\mathcal{F}^{-1} \mathcal{F} f$, hence they are the same.


**Multiplication and Differeniation** We had some identities about how $\mathcal{F}$ behaved with respect to multiplication by $x$ and differentiation. While this two operations gave linear operators on $\mathcal{S}$, they definitely do not extend to $L^2$, unlike $\mathcal{F}$. But when they do, they work as expected. More explicitly, suppose that $f \in L^2$ and the function $xf(x)$ is also in $L^2$. If $f_n \in \mathcal{S}$ approximate $f$ in $L^2$, we hope that $xf_n(x) \rightarrow xf(x)$ in $L^2$. But this is sadly not the case! We'd like to upgrade $f_n$ to do multiple things at once: have $f_n$ and $xf_n$ both converage at the same time!




todo
- approximate identities
- convolution
- "uncertainty princples"



# L1

- absolute convergence of hatf
- fhat is bounded, so only L1->Linfty
- fhat is uniformly continous
- Riemann-Lebesgue lemma
- convolution theorem
- translation and modulation
- differentiation rule works if f, xf in L1 => shows decay of f implies smoothness of fhat
- IF fhat in L1, then fhathat = f a.e.
- the indicator is in L1, but fourier is not, so inversion is hard
- exponential decay and indicator will be the mascots


coming soon

# measures

coming soon

# tempereted distributions

coming soon

# general distributions

coming soon


# Periodic Functions

Let's start a nice warmup with some very simple sorts of functions.

A function $f$ is *periodic with period $T$* if for all $t \in \mathbb{R}, n \in \mathbb{Z}$, we have $f(t + nT) = f(t)$.

Ok, great. If two functions $f,g$ have the same period $T$, then clearly $f+g$ has period $T$ and $cf$ has period $T$ if $c \in \mathbb{R}$. Various spaces of functions with the same fixed period will be interesting to us in a moment. But for now, we want to have a little more freedom. How about adding functions with different periods?. One easy case:

If $f$ has period $T_f$ and $g$ has period $T_g$, and $T_f/T_g$ is rational, then $f+g$ is periodic. If we write $T_f/T_g = n_g / n_f$ as a rational, then $f+g$ has period $T_{f+g} = n_f T_f = n_g T_g$, which is easy enough to see. But what if the ratio isn't rational? Well... you're in a little bit of trouble. Consider $\cos(t) + \cos(\sqrt{2} t)$. when $t = 0$, this is $2$. Is it ever $2$ again? We'd need $\cos(t) = 1 = \cos(\sqrt{2}t)$ for that to happen, which means that $t = 2\pi n$ for some integer $n$, but also $t = \frac{2\pi m}{\sqrt{2}}$ for some other integer $m$, and... you see the problem. So if we want to be able to do interesting things with periodic functions with different periods, we'll need to relax our requirements.

One thing we might notice is that while our countersample $\cos(t) + \cos(\sqrt{2} t)$ never reaches 2 again, it gets pretty close! In fact, by picking rational numbers closer and closer to $\sqrt{2}$, you can build points that get close to 2. So here is one attempt to generalize things:

A function is *almost periodic* if for every $\varepsilon > 0$, there is some $L$ so that every closed interval of length $L$ contains $\tau$ such that $\sup_t |f(t + \tau) - f(t) | < \varepsilon$. Clearly if $f$ is periodic with period $T$, then we can take $L = T$ for every $\varepsilon$.

The set of almost periodic function is a vector space. Suppose that $f, g$ are both almost periodic but $f+g$ is not, and there is some $\varepsilon_0$ so that the set $$\mathcal{T} := \bigg\\{ \tau: \sup_t |f(t+\tau) + g(t+\tau) - f(t) - g(t)| < \varepsilon_0 \bigg\\}$$ is not within a finite distance of every point. In other words, there must be points $x_n$ such that $dist(x_n, \mathcal{T}) \geq n$ for all $n$.



