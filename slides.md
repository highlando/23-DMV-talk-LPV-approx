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

---

## SDRE series expansion

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

**Praxis**
We use $\Pi(x)$ that solely solves the SDRE
$$
\Pi(x) A(x)+A^T(x) \Pi(x)-\frac{1}{\alpha} \Pi(x) BB^T\Pi(x)+C^TC=0,
$$
to compute the feedback.

# How to Design an LPV approximation

 * Under mild conditions, the flow $f(x)$ can be factorized
$$
\dot x = [A(x)]\,x
$$ 
with some $A\colon \mathbb R^{n} \to \mathbb R^{n\times n}$.

 * Trivially, this is an LPV representation with $\rho(x) = x$.

---

 * Any MOR scheme that compresses ($\mathcal P$) the state and lifts ($\mathcal L$) it back
 $$
 \tilde x = \mathcal L(\hat x) = \mathcal L (\mathcal P(x)) \approx x
 $$

. . .

 * gives a low-dimensional LPV approximation by means of $\rho = \mathcal P(x)$ and
 $$
 \dot x = A(x)\,x \approx A(\tilde x)\, x = A(\mathcal L \rho (x))\,x.
 $$

. . .

 * **Observation**: 
   * If $x\mapsto A(x)$ is linear 
   * and $\mathcal L$ is linear, 
   * then this LPV approximation is **linear**.


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
