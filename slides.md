---
author: 
 - Jan Heiland & Peter Benner (MPI Magdeburg)
title: Convolutional AEs for low-dimensional parameterizations of Navier-Stokes flow
subtitle: CSC Ringberg Workshop -- 2022
title-slide-attributes:
    data-background-image: pics/mpi-bridge.gif
parallaxBackgroundImage: pics/csc-en.svg
parallaxBackgroundSize: 1000px 1200px
bibliography: nn-nse-ldlpv-talk.bib
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
 * data-driven surrogate models
 * all of it?!


# LPV Representation

$$
\dot x = f(x) \approx A(x)\,x \approx [A_0+\Sigma \,\rho_k(x)A_k]\, x
$$

---

The *linear parameter varying* (LPV) representation/approximation
$$
\dot x = \approx  [A_0+\Sigma \,\rho_k(x)A_k]\, x + Bu
$$
with **affine parameter dependency** can be exploited for designing nonlinear controller through scheduling.

However, the dimension of $\rho$ should be very small.

---

## How to Design an LPV approximation

 * Under mild conditions, the flow $f(x)$ can be factorized
$$
\dot x = [A(x)]\,x
$$ 
with some $A\colon \mathbb R^{n} \to \mathbb R^{n\times n}$.

 * Trivially, this is an LPV representation with $\rho(x) = x$.

---

 * Any MOR scheme that compresses the state and *POD* and lifts it back
 $$
 \tilde x = \mathcal L(\hat x) = \mathcal L (\mathcal P(x))
 $$
 * gives a low-dimensional LPV approximation by means of $\rho = \mathcal P(x)$ and
 $$
 \dot x = A(x)\,x \approx A(\tilde x)\, x = A(\mathcal L \rho (x))\,x.
 $$
 * **Observation**: If $x\mapsto A(x)$ is linear as is $\mathcal L$, then this LPV approximation is **linear**.

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
be a, say, *POD* basis with $$v(t)=\tilde v(t) =\approx VV^Tv(t),$$

* then $$\rho(v(t)) = V^Tv(t)$$ is a parametrization.

---

* And with $$\tilde v = VV^Tv = V\rho = \sum_{k=1}^rV_k\rho_k,$$

* the NSE has the low-dimensional LPV representation via
$$
(v\cdot \nabla) v \approx (\tilde v \cdot \nabla) v = [\sum_{k=1}^r\rho_k(V_k\cdot \nabla)]\,v.
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
 * *Taylor-Hood* finite elements with over 60000 degrees of freedom
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
   * one linear layer $\rho \to \tilde \rho$
 * variable `code size` -- dimension of the parametrization $\rho$

:::

## {data-background-image="pics/dlvst-cs3.svg" data-background-size="100%"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Simulation parameters:

 * The nonlinearity $(v\cdot \nabla)v$ of the NSE
 * is replaced by $$\bigl[\frac 12 (W\rho \cdot \nabla)v +  (v\cdot \nabla)W\rho\bigr]$$
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
