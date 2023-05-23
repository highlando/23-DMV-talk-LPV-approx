---
author: 
 - Jan Heiland & Peter Benner (MPI Magdeburg)
title: Low-dimensional linear parameter varying system approximations for nonlinear controller design
subtitle: Blacksburg -- May 2023
title-slide-attributes:
    data-background-image: pics/mpi-bridge.gif
parallaxBackgroundImage: pics/csc-en.svg
parallaxBackgroundSize: 1000px 1200px
bibliography: nn-nse-ldlpv-talk.bib
nocite: |
  @*
---

# Introduction 

$$\dot x = f(x) + Bu$$

---

## {data-background-video="pics/triple_swingup_slomo.MP4"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Control of an inverted pendulum

 * 9 degrees of freedom
 * but nonlinear controller.

:::

## {data-background-image="pics/dbrc-v_Re50_stst_cm-bbw.png"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Stabilization of a laminar flow

 * 50'000 degrees of freedom
 * but linear regulator.

:::

## Control of Nonlinear & Large-Scale Systems

A general approach would include

 * powerful backends (linear algebra / optimization)
 * exploitation of general structures
 * MOR
 * data-driven surrogate models
 * all of it?!


# LPV Representation

\begin{align}
\dot x & = f(x) + Bu \\
       & \approx [A_0+\rho_1(x)A_1+ \dotsm + \rho_r(x) A_r]\, x + Bu
\end{align}

---

The *linear parameter varying* (LPV) representation/approximation
$$
\dot x \approx  \bigl [\Sigma \,\rho_i(x)A_i \bigr]\, x + Bu
$$
for nonlinear controller calls on

 * a general structure (linear(!) but parameter-varying)
 * model order reduction (to reduce the parameter dimension)

and on extensive theory

 1. LPV controller design
 2. series expansions of state-dependent Riccati equations

---

## LPV system approaches

For linear parameter-varying systems
$$
\dot x = A(\rho(x))\,x + Bu
$$
there exist established methods that provide control laws based one

 * robustness against parameter variations (REFREF)
 * adaption with the parameter (*gain scheduling*, ApKetal)

A major issue: require solutions of coupled LMI systems.


# SDRE series expansion

. . .

Consider the optimal control problem

$$
\int_0^\infty y^Ty + \alpha u^Tu ds
$$
subject to 
$$
\dot x = A(\rho(x))\,x+Bu, \quad y=Cx.
$$

---

**Theorem**

If there exists $\Pi$ as a function of $x$ such that
$$
\begin{aligned}
& \dot{\Pi}(x)+\bigl[\frac{\partial(A(x))}{\partial x}\bigr]^T \Pi(x)\\
& \quad+\Pi(x) A(x)+A^T(x) \Pi(x)-\frac{1}{\alpha} \Pi(x) BB^T \Pi(x)+C^TC=0 .
\end{aligned}
$$

Then $u=-\frac{1}{\alpha}B^T\Pi(x)\,x$ is an optimal feedback for the control problem.

---

**Praxis**
Parts of the HJB are discarded and we use $\Pi(x)$ that solely solves the state-dependent Riccati equation (SDRE)
$$
\Pi(x) A(x)+A^T(x) \Pi(x)-\frac{1}{\alpha} \Pi(x) BB^T\Pi(x)+C^TC=0,
$$
and the SDRE feedback
$$
u=-\frac{1}{\alpha}B^T\Pi(x)\,x.
$$

* numerous application examples [refrefref]
* proofs of performance []
* also beyond smallness conditions [BenH]

## The series expansion

We note that $\Pi$ depends on $x$ through $A(\rho(x))$. 

Thus, we can consider $\Pi$ as a function in $\rho$ and its corresponding multivariate Taylor expansion

* $\alpha=(\alpha_1, \dotsc, \alpha_r)$ is a multiindex and
* $\Pi_{(\alpha)}$ are **constant** matrices

---

If we insert the Taylor expansion of $\Pi$ and the LPV representation of $A$:

by *matching the coefficients*, we obtain equations for the matrices of, say, the first order approximation

---

Thus, the first-order approximation (in $\rho$) is obtained as

where $P_0$ solves 

and $L_k$ solve 

for $k=1,\dotsc,r$.

...

And the nonlinear feedback is realized as
$$
u = -\frac{1}{\alpha}B^T[P_0 + \sum_{i=k}^r \rho_k(x) L_k]\,x
$$

# How to Design an LPV approximation

A general procedure

---

If $f(0)=0$ and under mild conditions, the flow $f$ can be factorized
$$
f( x) = [A(x)]\,x
$$ 
with some $A\colon \mathbb R^{n} \to \mathbb R^{n\times n}$.

. . .

1. If $f$ has a strongly continuous Jacobian $\partial f$, then
$$
f(x) = [\int_0^1 \partial f(sx)\mathsf{d} s]\, x
$$
2. The trivial choice of
$$
f(x) = [\frac{1}{x^Tx}f(x)x^T]\,x
$$
doesn't work well (neither do the improvements [Charles]).


---

For the factorization $f(x)=A(x)\,x$, one can say that

1. it is not unique
2. it can be a design parameter
3. often, it is indicated by the structure.

. . .

... like in the advective term in the *Navier-Stokes* equations:
$$
(v\cdot \nabla)v = \mathcal A_s(v)\,v
$$
with $s\in[0,1]$ and the linear operator $\mathcal A_s(v)$ defined via 
$$\mathcal A_s(v)\,w := s\,(v\cdot \nabla)w + (1-s)\, (w\cdot \nabla)v.$$

---

## $\dot x = A(x)\,x + Bu$

 * Trivially, this is a (quasi) LPV representation 
 $$
 \dot x = A(\rho(x))\, x + Bu
 $$
 with $\rho(x) = x$.

 * Take any MOR scheme that compresses (via $\mathcal P$) the state and lifts it back (via $\mathcal L$) so that
 $$
 \tilde x = \mathcal L(\hat x) = \mathcal L (\mathcal P(x)) \approx x
 $$

. . .

 * Then $\rho = \mathcal P(x)$ gives a low-dimensional LPV approximation by means of
 $$
 A(x)\,x \approx A(\tilde x)\, x = A(\mathcal L \rho (x))\,x.
 $$

---

## Observation

   * If $x\mapsto A(x)$ itself is affine linear 
   * and $\mathcal L$ is linear, 
   * then
   $$
   \dot x \approx A(\mathcal L \rho(x))\,x + Bu = [A_0 + \sum_{i=1}^r \rho_i(x) A_i]\, x + Bu
   $$
   is affine with 

     * $\rho_i(x)$ being the components of $\rho(x)\in \mathbb R^r$ 
     * and constant matrices $A_0$, $A_1$, ..., $A_r \in \mathbb R^{n\times n}$.


# Numerical Realization

## {data-background-image="pics/cw-Re60-t161-cm-bbw.png" data-background-size="cover"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}
The *Navier-Stokes* equations

$$
\dot v + (v\cdot \nabla) v- \frac{1}{\mathsf{Re}}\Delta v + \nabla p= f, 
$$

$$
\nabla \cdot v = 0.
$$
:::

---

## {data-background-image="pics/cw-Re60-t161-cm-bbw.png" data-background-size="cover"}

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}
Control Problem:

 * use two small outlets for fluid at the cylinder boundary
 * to stabilize the unstable steady state
 * with a few point observations in the wake.

:::

---

## {data-background-image="pics/cw-Re60-t161-cm-bbw.png" data-background-size="cover"}

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}
Simulation model:

 * we use *finite elements* to obtain
 * the dynamical model of type

 $\dot x = Ax + N(x,x) + Bu, \quad y = Cx$

 * with $N$ being bilinear in $x$
 * and a state dimension of about $n=50'000$.

:::

---

## The Algorithm

Nonlinear controller design for 
$$
\dot x = f(x) + Bu
$$
by LPV approximations and truncated SDRE expansions.

. . .

1. Compute an affine LPV approximative model with 
$$f(x)=[\sum_{k=0}^r \rho_k(x)A_k]\,x.$$

2. Solve one *Riccati* and $r$ *Lyapunov* equations for $P_0$ and the $L_k$s.
3. Close the loop with $u = -\frac{1}{\alpha}B^T[P_0 + \sum_{k=1}^r \rho_k(x) L_k]\,x.$

## 1 Compute the LPV Approximation

We use POD coordinates with the matrix $V\in \mathbb R^{n\times r}$ of POD modes $v_k$

 * $\rho(x) = V^T x$, 

 * $\tilde x = V\rho(x)=\sum_{k=1}^r\rho_i(x)v_k.$

. . .

Then:
$$N(x,x)\approx N(\tilde x, x) = N(\sum_{k=1}^r\rho_i(x)v_k, x) = \sum_{k=1}^r\rho_i(x) N(v_k, x) $$
which is readily realized as
$$ [\sum_{k=1}^r\rho_i(x) A_k]\,x.$$

## 2 Compute $P_0$ and the $L_k$s

This requires the solve of large-scale ($n=50'000$) matrix equations

1. Riccati -- nonlinear but fairly standard
2. Lyapunovs -- linear but indefinite.

We use state-of-the-art low-rank ADI iterations (ask Steffen for details).


---

## {data-background-image="pics/cw-v-Re60-stst-cm-bbw.png" data-background-size="cover"}

**3 Close the Loop**

# Conclusion

## ... and Outlook

 * LPV with affine-linear dependencies are attractive if only $k$ is small.

 * CNNs can provide such very low dimensional LPV approximations 

 * and clearly outperform POD (at very low dimensions).

 * Lots of potential left for further improvement.

 * Outlook: Use for nonlinear controller design.

. . .

Thank You!

---

<!--

# Low-dimensional LPV for NSE

**LPV Approximation** of *Navier-Stokes Equations* by *POD* and *Convolutional Neural Networks*

---


## {data-background-image="pics/cw-Re60-t161-cm-bbw.png" data-background-size="cover"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}
The *Navier-Stokes* equations

$$
\dot v + (v\cdot \nabla) v- \frac{1}{\mathsf{Re}}\Delta v + \nabla p= f, 
$$

$$
\nabla \cdot v = 0.
$$
:::

---

* Let $v$ be the velocity solution and let
$$
V =
\begin{bmatrix}
V_1 & V_2 & \dotsm & V_r
\end{bmatrix}
$$
be a, say, *POD* basis with $$v(t)=\tilde v(t) \approx VV^Tv(t),$$

* then $$\rho(v(t)) = V^Tv(t)$$ is a parametrization.

---

* And with $$\tilde v = VV^Tv = V\rho = \sum_{i=1}^rV_i\rho_i,$$

* the NSE has the low-dimensional LPV representation via
$$
(v\cdot \nabla) v \approx (\tilde v \cdot \nabla) v = [\sum_{i=1}^r\rho_i(V_i\cdot \nabla)]\,v.
$$

## Question

Can we do better than POD?

## {data-background-image="pics/scrsho-lee-cb.png"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Lee/Carlberg (2019): *MOR of dynamical systems on nonlinear manifolds using deep convolutional autoencoders*
:::

## {data-background-image="pics/scrsho-choi.png"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Kim/Choi/Widemann/Zodi (2020): *Efficient nonlinear manifold reduced order model*
:::

## Convolution Autoencoders for NSE

1. Consider solution snapshots $v(t_k)$ as pictures.

2. Learn convolutional kernels to extract relevant features.

3. While extracting the features, we reduce the dimensions.

4. Encode $v(t_k)$ in a low-dimensional $\rho_k$.

## Our Example Architecture Implementation


## {data-background-image="pics/nse-cnn.jpg"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

 * A number of convolutional layers for feature extraction and reduction

 * A full linear layer with nonlinear activation for the final encoding $\rho\in \mathbb R^{r}$

 * A linear layer (w/o activation) that expands $\rho \to \tilde \rho\in \mathbb R^{k}$.

 * And $\tilde \rho$ is used as "parametrized POD coordinates"

:::

## Training for minimizing:
$$
\| v_i - VW\rho(v_i)\|^2_M + 
\| (v_i\cdot \nabla)v_i - (VW\rho_i \cdot \nabla )v_i\|^2_{M^{-1}}
$$
which includes

 1. the POD modes $V\in \mathbb R^{n\times k}$,

 2. a learned weight matrix $W\in \mathbb R^{k\times r}\colon \rho \mapsto \tilde \rho$,

 3. the mass matrix $M$ (and it's inverse) of the FEM discretization.

# Numerical Example


## {data-background-image="pics/example-cnn-design.png"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

 * Example architecture of the CNN encoder/decoder

 * Designed and optimized in `pytorch`

 * Very many parameters of the design and the optimization -- not yet well explored

:::

## {data-background-image="pics/sc-RE40-dvlpd.png"}


. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Simulation parameters:

 * Cylinder wake at $\mathsf{Re}=40$, time in $[0, 50]$
 * *Taylor-Hood* finite elements with over `60000` degrees of freedom
 * `2000` snapshots/data points on $[0, 8]$ for the POD and CNN

:::

## {data-background-image="pics/Figure_9.svg"}


. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

CNN parameters:

 * `2000` data points on the FEM mesh
 * interpolated to a tensor of size `2x63x127`
 * convolutional neural network 
   * convolutional layers
     * `kernelsize, stride = 5, 2`
   * one activated linear layer $\to \rho$
   * one linear layer $\rho \mapsto \tilde \rho$

:::

## {data-background-image="pics/dlvst-cs3.svg" data-background-size="100%"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Simulation parameters:

 * The nonlinearity $(v\cdot \nabla)v$ of the NSE is replaced by $$\frac 12 \bigl[(W\rho \cdot \nabla)v +  (v\cdot \nabla)W\rho\bigr]$$
 * with $\rho = \rho (v) \in \mathbb R^{3}$
 * encoded and decoded through
   * plain POD or
   * a CNN 

:::

## {data-background-image="pics/dlppt-cs3.svg" data-background-size="100%"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

The limit cycle:

 * We report the drag and lift forces that act on the cylinder
 * and plot their phase portrait.
 * The limit cycle is very well captured by the CNN of `code_size=3`
 * POD approximation far off

:::

# Outlook

Application in nonlinear controller design

---

For an LPV system
\begin{equation*}
\dot x = A(\rho)\, x + Bu,
\end{equation*}
controller design by *gain scheduling* bases on 

. . .

 1. identification of salient parameter configurations $\rho^{(k)}$ (*working points*) 

 1. detection of distances $d(\rho, \rho^{(k)})$ of the current configuration $\rho$

 1. interpolation of controllers designed for the *working points*

. . .

::: {style="position: absolute; width: 90%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

For an LPV representation of the Navier-Stokes equations this 

* can only be realized 
* by clustering
* in the, hopefully, very low-dimensional parametrization

:::


## {data-background-image="pics/dlppt-cs3.svg" data-background-size="100%"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 30px; text-align: left;"}

![Phase portrait and clustering of the first two principal components of $\rho$](pics/rho2d_dist.png)

:::
-->
