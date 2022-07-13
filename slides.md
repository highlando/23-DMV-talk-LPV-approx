---
author: 
 - Jan Heiland & Peter Benner (MPI Magdeburg)
title: Very LD parametrizations of fluid flow for nonlinear controller design
subtitle: ORCOS -- Vienna -- July 2022
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
 * data-driven surrogate models
 * all of it?!


# LPV Representation

$$
\dot x = f(x) \quad = A(x)\,x \approx [A_0+\Sigma \,\rho_k(x)A_k]\, x
$$

---

The *linear parameter varying* (LPV) representation/approximation
$$
\dot x \approx  \bigl [\Sigma \,\rho_i(x)A_i\bigr]\, x
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
