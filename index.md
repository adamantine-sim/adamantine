---
layout: page
title: Home
nav_order: 1
usemathjax: true
---

# Home

*adamantine* stands for **AD**ditive (**A**) **MAN**unifac**T**uring s**I**mulator (**NE**).

## Introduction
**adamantine** is a software for simulating additive manufacturing. It is based on
the [deal.II](https://www.dealii.org) library. **adamantine** can simulate the
heat field during the manufacturing process and the phase change that the
material undergoes. The addition of material is done using element activation.
Experimental data can be used to improve the simulation through the use of 
[Ensemble Kalman filter](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).

## Governing equations
### Assumptions
We make the following assumptions:
* No movement in the liquid.
* No evaporation of the material.
* No change of volume when the material changes phase.
* We assume that there is always a mushy zone (no isothermal change
  of phase).

### Heat equation
#### Weak form
The heat equation without phase change is given by:

$$
\rho(T) C_p(T) \frac{\partial T}{\partial t} - \nabla \cdot \left(k\nabla T\right) = Q,
$$  

where $\rho$ is the mass density, $$C_p$$ is the specific heat, $$T$$, is the
temperature, $$k$$ is the thermal conductivity, and $$Q$$ is the volumetric heat
source.

When there is a phase change, the heat equation is usually written in term of
the enthalpy, $$h$$:

$$
\frac{\partial h(T)}{\partial t} -  \nabla \cdot \left(k\nabla T\right) = Q
$$

In the absence of phase change, we have:

$$
h(T) = \int_{T_0}^T \rho(T) C_p(T) dT.
$$

Under the assumption of a phase change with a mushy zone, $$C_p$$ and $$\rho$$ are independent
of the temperature, we write:

$$
h(T) =
  \cases{
   \rho_s C_{p,s} T & if $T<T_{s}$\cr
   \rho_s C_{p,s} T_s + \left(\frac{\rho_s C_{p,s}+\rho_l C_{p,l}}{2} +
    \frac{\rho_s+\rho_l}{2}  \frac{\mathcal{L}}{T_l-T_s}\right) (T-T_s) & if $T>T_{s}$ and $T<T_l$ \cr
    \rho_s C_{p,s} T_s + \frac{C_{p,s}+C_{p,l}}{2} (T_l - T_s) +
    \frac{\rho_s+\rho_l}{2} \mathcal{L} + \rho_s C_{p,l}
    (T-T_l) & if $T>T_l$.
  }
$$

Since we only care about $$\frac{\partial h{T}}{\partial t}$$, we have:

$$
\frac{\partial h(T)}{\partial t} =
  \cases{
    \rho_s C_{p,s} \frac{\partial T}{\partial t} &  if $T \leq T_{s}$\cr
     \left(\rho_{\text{eff}} C_{p,\text{eff}} + \rho_{\text{eff}} \frac{\mathcal{L}}{T_l-T_s}\right)
     \frac{\partial T}{\partial t}  & if $T>T_{s}$ and $T<T_l$ \cr
    \rho_l C_{p,l} \frac{\partial T}{\partial t} &  if $T \geq T_{l}$
  }
$$

Note that we have a more complicated setup because we have two solid phase
(solid and powder).

So far we haven't discussed $$k$$. $$k$$ is simply given by:

$$
k =
  \cases{
    k_s & if $T \leq T_s$ \cr
    k_s + \frac{k_l - k_s}{T_l - T_s} (T- T_s) & if $T>T_s$ and $T<T_l$ \cr
    k_l & if $T \geq T_l$
  }
$$

Finally we can write:
* if $$T \leq T_s$$, we have:

  $$
  \frac{\partial T}{\partial t} = \frac{1}{\rho_s C_{p,s}} \left(\nabla \cdot \left(k
  \nabla T\right) + Q\right)
  $$

* if $$T_s < T < T_l$$, we have:

  $$
  \frac{\partial T}{\partial t} = \frac{1}{\left(\rho_{\text{eff}}
  C_{p,\text{eff}} + \rho_{\text{eff}} \frac{\mathcal{L}}{T_l-T_s}\right)} \left(
  \nabla \cdot \left(k \nabla T\right) + Q \right)
  $$

* if $$T \geq T$$, we have:

  $$
  \frac{\partial T}{\partial t} = \frac{1}{\rho_l C_{p,l}} \left(\nabla \cdot \left(k
  \nabla T\right) + Q\right)
  $$

Next, we will focus on the weak form of:

$$
\frac{\partial T}{\partial t} = \frac{1}{\rho C_{p}} \left(\nabla \cdot \left(k
\nabla T\right) + Q\right).
$$

We have succesively with $$\alpha = \frac{1}{\rho C_{p}}$$:

$$
\int b_i \frac{\partial T_i b_j}{\partial t} = \int b_i \alpha \left(\nabla \cdot \left(k
\nabla T_j b_j\right) + Q\right),
$$

$$
\int b_i b_j \frac{d T_j}{dt} = \int \alpha T_j b_i \nabla \cdot \left(k \nabla b_j\right) +
\int \alpha b_i Q,
$$

$$
\left(\int b_i b_j\right) \frac{d T_j}{dt} = - \int \alpha T_j \nabla b_i \cdot \left(k \nabla b_j\right) +
\int_{\partial} \alpha T_j b_i \boldsymbol{n}\cdot \left(k \nabla b_j\right) + \int \alpha b_i Q.
$$

#### Boundary Condition
We are now interested in the boundary term $$\int_{\partial} \alpha T_j b_i
\boldsymbol{n}\cdot \left(k \nabla b_j\right)$$, in the interest of understanding the
physical meaning of this term, we will write it as:

$$
\int_{\partial} \alpha b_i \boldsymbol{n}\cdot \left(k \nabla T\right)
$$

If this term is equal to zero, this means that $$\nabla T=0$$. Physically this
condition corresponds to a reflective boundary condition. In practice, we are
interested in two kind of boundary conditions: radiative loss and convection.
##### Radiative Loss
The Stefan-Boltzmann law describes the heat flux due to radiation as:

$$
-\boldsymbol{n} \cdot  \left(k \nabla T\right) = \varepsilon \sigma \left(T^4 -T_{\infty}^4\right),
$$

with $$\varepsilon$$ the emissitivity and $\sigma$ the Stefan-Boltzmann constant.
The value of $$\sigma$$ is (from NIST):

$$
\sigma = 5.670374419 \times 10^{-8} \frac{W}{m^2 k^4}.
$$

We can write:

$$
\begin{split}
  \int_{\partial} \alpha b_i \boldsymbol{n} \cdot \left(k\nabla T\right) &=
  -\int_{\partial} \alpha b_i \varepsilon \sigma \left(T^4 - T_{\infty}^4\right),\\\\\\
  &= -\int_{\partial} \alpha b_i \varepsilon \sigma T^4 +
  \int_{\partial} \alpha b_i \varepsilon T_{\infty}^4
\end{split}
$$

We can now use this equation to impose the radiative loss. However,
this is nonlinear. Thus, we need to use a Newton solver to impose
the boundary condition. This is less than ideal. Instead, we will linearize the
Stefan-Boltzmann equation:

$$
-\boldsymbol{n} \cdot \left(k\nabla T\right) = h_{\text{rad}}\left(T-T_{\infty}\right),
$$

with

$$
h_{\text{rad}} = \varepsilon \sigma\left(T+T_{\infty}\right)\left(T^2 + T_{\infty}^2\right).
$$

Thus, we have:
$$
\begin{split}
  \int_{\partial} \alpha b_i \boldsymbol{n} \cdot \left(k \nabla T\right) &=
  -\int_{\partial} \alpha b_i h_{\text{rad}} \left(T-T_{\infty}\right),\\\\\\
  &=-\int_{\partial} \alpha h_{\text{rad}} \sum_j T_j b_i b_j +
  \int_{\partial} \alpha h_{\text{rad}} T_{\infty} b_i.
\end{split}
$$

##### Convection
The convective heat transfer has the same form as the linearized radiative loss:

$$
-\boldsymbol{n} \cdot \left(k\nabla T\right) = h_{\text{conv}}\left(T-T_{\infty}\right).
$$

Thus, we have:

$$
\begin{split}
  \int_{\partial} \alpha b_i \boldsymbol{n} \cdot \left(k \nabla T\right) &=
  -\int_{\partial} \alpha b_i h_{\text{conv}} \left(T-T_{\infty}\right),\\\\\\
  &=-\int_{\partial} \alpha h_{\text{conv}} \sum_j T_j b_i b_j +
  \int_{\partial} \alpha h_{\text{conv}} T_{\infty} b_i.
\end{split}
$$

## Algorithmic choice
### Matrix-free implementation
The implementation is done matrix-free for the following reasons:
* New architecture have little memory per core and so not having to store
    the memory is very interesting.
* Because the latency of the memory, a very important part of our problem
    is memory bound. It is therefore interesting to decrease memory access even
    at the cost of more computation.
* Because we have time-dependent nonlinear problem, we would need to
    rebuild the matrix at least every time step. Since the assembly needs to be
    redone so often, storing the matrix is not advantageous.

### Adaptive mesh refinement
Usually, the powder layer is about 50 microns thick but the piece that is being
built is several centimeters long. Moreover, since the material is melted using
an electron beam or a laser, the melting zone is very localized. This means that
a uniform would require a very large number of cells in place where nothing
happens (material not heated yet or already cooled). Using AMR, we can refine
the zones that are of interest for during a given time.

## Data assimilation
The goal of data assimilation is to combine a numerical simulation simulation.
Experimental data from infra-red cameras and thermo-couples can
be used to improve the simulation. This is done using  the stochastic Ensemble 
Kalman filter method.
