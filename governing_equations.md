---
layout: default
parent: Home
title: Governing Equations
nav_order: 1
usemathjax: true
---

# Governing Equations
## Assumptions
We make the following assumptions:
1. No movement in the liquid.
2. No evaporation of the material.
3. No change of volume when the material changes phase.
4. There is always a mushy zone (no isothermal change of phase).
5. One-way coupling from the temperature evolution to the mechanical evolution. We neglect 
the effect of deformation on the thermal simulation.
6. Material properties used in the heat equation can be anisotropic
7. Material properties used in the solid mechanics are isotropic

## Heat equation
### Weak form
The heat equation without phase change is given by:

$$
\rho(T) C_p(T) \frac{\partial T}{\partial t} - \nabla \cdot \left(k\nabla T\right) = Q,
$$  

where $$\rho$$ is the mass density, $$C_p$$ is the specific heat, $$T$$, is the
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

### Boundary Condition
We are now interested in the boundary term $$\int_{\partial} \alpha T_j b_i
\boldsymbol{n}\cdot \left(k \nabla b_j\right)$$, in the interest of understanding the
physical meaning of this term, we will write it as:

$$
\int_{\partial} \alpha b_i \boldsymbol{n}\cdot \left(k \nabla T\right)
$$

If this term is equal to zero, this means that $$\nabla T=0$$. Physically this
condition corresponds to a reflective boundary condition. In practice, we are
interested in two kind of boundary conditions: radiative loss and convection.
#### Radiative Loss
The Stefan-Boltzmann law describes the heat flux due to radiation as:

$$
-\boldsymbol{n} \cdot  \left(k \nabla T\right) = \varepsilon \sigma \left(T^4 -T_{\infty}^4\right),
$$

with $$\varepsilon$$ the emissitivity and $$\sigma$$ the Stefan-Boltzmann constant.
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

#### Convection
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

## Thermomechanical equations
### Thermoelasticity
Because of *assumption 5*, the thermal equation described in the previous
section is unchanged. To this equation, we need to add the solid mechanics
equations. Using *assumption 7* and Einstein summation convention, we get the following[^1]:

 - the Cauchy momentum equation:

$$
\frac{D v_i}{Dt} = \frac{1}{\rho} \frac{\partial \sigma_{ij}}{\partial x_j} + f_i
$$

 - the strain-displacement equations:

$$
\varepsilon_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial u_j} + \frac{\partial u_j}{\partial u_i}\right)
$$ 

 - the constitutive equations:

$$
\sigma_{ij} = \lambda \varepsilon_{\nu \nu} \delta_{ij} + 2 \mu \varepsilon_{ij} -\beta \delta_{ij}
\theta,
$$

with 

$$
\beta = (3 \lambda + 2 \mu) \alpha
$$

where $$\boldsymbol{v}$$ is the flow velocity vector, $$\frac{D v_i}{Dt} = \frac{\partial v_i}{\partial t} + v_j \frac{\partial v_i}{\partial x_j}$$ is the material derivative, $$\rho$$ is the material density, $\boldsymbol$ is the volumetric force, $$\sigma$$ is the stress tensor, $$\varepsilon$$ is the infinitesimal strain tensor, $$\boldsymbol{u}$$ is the displacement vector, $$\alpha$$ is the thermal coefficient of linear expansion, $$\lambda$$ is Lamé first parameter, $$\mu$$ is Lamé second parameter and $$\theta = T - T_0$$, $$T$$ is the current temperature, and $$T_0$$ is a reference temperature.

[^1]: A detailed discussion on thermoelasticity is available in Chapter 14 of *Classical and Computational Solid Mechanics* by Y.C. Fung & Pin Tong.

### Elastoplasticity
We implement the radial return algorithm for $$J_2$$ theory with a linear combination of isotropic and kinematic hardening described in Chapter 3 of *Plasticity Modeling & Computation* by Ronaldo I. Borja. The algorithm works as follow:

 1. Compute successively

$$
\sigma_{n+1}^{tr} = \sigma_{n} + c^e : \Delta \varepsilon,
$$

$$
p = \frac{1}{3} \text{tr}\left(\sigma_{n+1}^{tr}\right),
$$

$$
s_{n+1}^{tr} = \sigma_{n+1}^{tr}-p \boldsymbol{1},
$$

$$
\xi_{n+1}^{tr} = s_{n+1}^{tr} - \gamma_{n}
$$

where the subscript $$n$$ indicates the $$n$$ time step, the superscript $$tr$$
indicates trial, $$c^e$$ is the tensor of elastic moduli, $$\Delta \varepsilon$$
is the incremental strain, $$p$$ is the mean normal stress, $$s$$ is the
deviatoric stress tensord, and $$\gamma_{n}$$ is the back stress tensor.

 2. Compute 
$$
\chi = ||\xi_{n+1}^{tr}||
$$

 3. - If $$\chi \leq \kappa_{n}$$, set $$\sigma_{n+1} = \sigma_{n+1}^{tr}$$.
    - If $$\chi > \kappa_{n}$$, set 

$$
n_{n+1} = \frac{\xi_{n+1}^{tr}}{\chi},
$$

$$
\Delta \eta = \frac{\chi - \kappa_{n}}{2 \mu + H},
$$

$$
\sigma_{n+1} = \sigma_{n+1}^{tr} - 2 \mu \Delta \eta n_{n+1},
$$

$$
\kappa_{n+1} = \kappa_{n} + a H \Delta \eta,
$$  

$$
\gamma_{n+1} = \gamma_{n} + (1-a) H \Delta\eta n_{n+1}
$$

where $$\kappa_{n}$$ is plastic internal variable, $$H$$ is generalized plastic modulus, and $$a$$ is coefficient between 0 (no isotropic hardening) and 1 (no kinematic hardening).

### Thermoelastoplasticity
The thermoelastoplastic model implemented is the union of the thermoelastic model
and the elastoplastic model.
