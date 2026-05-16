# Schwartz functions

The first step on our Fourier analysis journey will be the space of Schwartz functions, on which the Fourier transform has pretty much all the nice properties you could possibly want.

# Definition

Suppose that $f:\mathbb{R} \rightarrow \mathbb{C}$ is a smooth (infinitely differentiable) function. For $n,m$ nonnegative integers, we can define a norm $$\\|f \\|_\{n,m\} = \sup\_{x \in \mathbb{R}} |x^n f^{(m)}(x)| $$

and from this define the *Schwartz space* $\mathcal{S}$ of all such functions for which all of these norms are finite:

$$ \mathcal{S} := \\{ f \in C^{\infty}(\mathbb{R}) : \forall_n \forall_m \\|f \\|_\{n,m\} < \infty \\} $$

These are very nice functions - smooth, with strong decay properties - that allow a nice development of Fourier analysis. Some facts that are easy to see from the definition:

* $\mathcal{S}$ is closed under differentiation and multiplication by polynomials
* If $f,g \in \mathcal{S}$, so is $fg$.

## Examples 
Once nice class of functions in $\mathcal{S}$ are smooth compactly supported functions $C_c^\infty$ - the functions $f$ which have an infinite number of derivatives, and also some interval $N$ so that $f(x) = 0$ if $|x| > N$. 

Another class of functions in $\mathcal{S}$ are functions of the form $e^{-f(x)}$ where $f$ has no worse than polynomial growth. For instance, the PDF of the Gaussian $e^{-x^2}$ is an example.


# $\mathcal{F}$ and basic properties

We can now define the Fourier transform and its inverse:

<div class='def'>

$$
\begin{align*}
\mathcal{F}\[f\](\omega) &= \frac{1}{\sqrt{2\pi}} \int f(t) e^{-\omega i t} dt \\\\
\mathcal{F}^{-1}\[\hat{f}](t) &= \frac{1}{\sqrt{2\pi}} \int \hat{f}(\omega) e^{\omega i t} d\omega
\end{align*}
$$

</div>


I'll sometimes write $\hat{f}$ for $\mathcal{F}[f]$ or $\check{f}$ for $\mathcal{F}^{-1}[f]$. 

<div class="thm">

**Theorem** If $f \in \mathcal{S}$, then both $\mathcal{F}\[f\]$ and $\mathcal{F}^{-1}\[f\]$ exist.

</div>

<div class="proof">

**Proof** I'll just stick to showing everything for $\hat{f} = \mathcal{F}[f]$. The inverse follows almost exactly the same way. 

Since $|x^2 f(x)| \leq C$ is finite, we get $$\int |f(x) e^{- i \omega x}| dt \leq \int_{-1}^1 |f(x)| dx + \int_{[-1,1]^c} \frac{C}{|x|^2} dx$$ converges absolutely, so $\hat{f}$ exists. Note this bound does not depend on $\omega$. ☐

</div>

<div class="thm">

**Theorem** If $f \in \mathcal{S}$, then

$$
\begin{align*}
\mathcal{F}\[f^{(n)}\](\omega) &= (i \omega)^n \mathcal{F}\[f\](\omega) \\\\
\mathcal{F}\[x^m f\](\omega) &= i^m \frac{d^m}{d\omega^m} \mathcal{F}\[f\](\omega)
\end{align*}
$$

That is, the Fourier transform turns mulplication by $x$ into derivatives and derivatives into multiplication by $\omega$. This is a vital property of $\mathcal{F}$, and makes it an extremely useful tool for solving differential equations, since it turns them into algebraic ones.

</div>

<div class="proof">

**Proof** 

For the first equality, integrating by parts gives
$$\widehat{f'}(\omega) = \frac{1}{\sqrt{2\pi}}\int f'(x) e^{-ix\omega} dx = \frac{i\omega}{\sqrt{2\pi}} \int f(x) e^{-ix\omega}dx = i\omega \hat{f}(\omega)$$ and so $$\widehat{f^{(n)}}(\omega) = (i\omega)^n \hat{f}(\omega).$$

For the second equality, we can differentiate under the integral sign: $- i x f(x) e^{- i \omega x}$ is the partial derivative of the integrand of $\mathcal{F}$ wrt $\omega$, which is bounded by $\frac{1}{|x|^2}$ at the tails and continuous near 0, and so $$\frac{d}{d\omega} \hat{f}(\omega) = \frac{1}{\sqrt{2\pi}} \int -i x f(x) e^{- i \omega x} dx = -i \widehat{xf}(\omega)$$ ☐
</div>

<div class="thm">

**Theorem** If $f \in \mathcal{S}$, then $\hat{f} \in \mathcal{S}$. In other words, $\mathcal{F}$ defines a function $$\mathcal{F} : \mathcal{S} \rightarrow \mathcal{S}.$$
</div>


<div class = "proof">

If $f \in \mathcal{S}$, we already showed that $\\\| \hat{f} \\\|_{0,0}$ is finite when we showed that $\mathcal{F}[f]$ is defined. The previous theorem shows that the derivatives of $\hat{f}$ exist, and 

$$ \\\| \hat{f} \\\|\_{n,m} = \bigg\\\| \omega^n \frac{d^m}{d\omega^m} \hat{f} \bigg\\\|_{0,0} < \infty$$ ☐
</div>

Here is an interesting fact - recall that the Gaussian is in $\mathcal{S}$. We can compute, if $g$ is the Gaussian function $\frac{1}{\sqrt{2\pi}}e^\frac{-x^2}{2}$...

$$\mathcal{F}[g] = \frac{1}{\sqrt{2\pi}} \int \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} e^{-it\omega} dt =  \frac{1}{2\pi} e^{-\frac{\omega^2}{2}} \int  e^{-\frac{(t + i \omega)^2}{2}} dt = g(\omega),$$ where we use the fact that $\sqrt{2\pi} = \int e^{-\frac{\omega}{2}^2} d\omega$ works along any line parallel to the real axis. Hence the Gaussian is an *eigenfunction* for $\mathcal{F}$!

<div class = "theorem">

**Theorem** $\mathcal{F}^{\pm 1}$ are inverses on $\mathcal{S}$.

</div>

<div class = "proof">

**Proof** Let's start with $\mathcal{F}^{-1} \mathcal{F}$. We need to compute

$$ \frac{1}{2\pi} \int \int f(t) e^{-i\omega t} e^{i \omega x} dt d\omega $$

We want to apply Fubini and compute $$\int e^{i\omega(x-t)} d\omega,$$ but this is actually a funny integral. Use finite bounds and assume $x \neq t$ and you get $$\int_{-L}^L e^{i\omega(x-t)} d\omega = \frac{e^{i\omega (x-t)}}{i(x-t)} \bigg|_{-L}^L = \frac{e^{iL(x-t)} - e^{-iL(x-t)}}{i(x-t)} = \frac{2\sin(L(x-t))}{x-t}$$ When $x = t$, the integral is just $2L$. In either case, it's clear that taking $L \rightarrow \infty$ is going to cause some problems, which is some foreshadowing of things to come. However, we can be sneaky. 

The idea is that we will integrate the double integral over a carefully crafted region that makes the integrals simple. It starts by noting that when $L = \frac{2\pi k}{x-t}$, this troublesome integral is just 0. So we can build regions $B_n(t)$ defined as

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

</div>

The last theorem uses the following dot product on $\mathcal{S}$:

$$\langle f, g \rangle = \int f(x) \overline{g(x)} dx$$ with norm $\\| f \\|_2 = \sqrt{\langle f, f \rangle}.$

<div class = "theorem">

**Theorem** (Plancherel's identity) $$ \\| f \\|_2 = \\|\hat{f} \\|_2$$

</div>

<div class = "proof">

**Proof** The nice way to do this is, recalling that $\check{g}$ is the inverse Fourier transform of $g$,

$$
\begin{align*}
&\langle f, \check{g} \rangle \\\\
= &\int f(x) \overline{\check{g}(x)} dx \\\\
= &\frac{1}{\sqrt{2\pi}} \int \int f(x) \overline{g(\omega)} e^{-i\omega x} d\omega dx \\\\
= &\frac{1}{\sqrt{2\pi}} \int \hat{f}(\omega) g(\omega) d \omega \\\\
= &\langle \hat{f}, g \rangle
\end{align*}
$$

Hence $$\\| f\\|^2_2 = \langle f, f \rangle = \langle \check{\hat{f}}, f \rangle = \langle \hat{f}, \hat{f} \rangle = \\| \hat{f} \\|^2_2.$$☐

</div>

