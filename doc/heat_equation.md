---
layout: default
parent: Home
nav_order: 4
published: false
usemathjax: true
---

# Heat equation
## Weak form
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

We have successively with $$\alpha = \frac{1}{\rho C_{p}}$$:

$$
\int b_i \frac{\partial T_j b_j}{\partial t} = \int b_i \alpha \left(\nabla \cdot \left(k
\nabla T_j b_j\right) + Q\right),
$$

$$
\int b_i b_j \frac{d T_j}{dt} = \int T_j \alpha b_i \nabla \cdot \left(k \nabla b_j\right) +
\int \alpha b_i Q,
$$

$$
\left(\int b_i b_j\right) \frac{d T_j}{dt} = - \int T_j \nabla \left(\alpha b_i\right) \cdot \left(k \nabla b_j\right) +
\int_{\partial} \alpha T_j b_i \boldsymbol{n}\cdot \left(k \nabla b_j\right) + \int \alpha b_i Q.
$$

The term $$-\int T_j \nabla \left(\alpha b_i\right) \cdot \left(k \nabla b_j\right)$$ can be written as:

$$\\
\begin{align}
-\int T_j \nabla \left(\alpha b_i\right) \cdot \left(k \nabla b_j\right) 
  &= -\int T_j \left(\left(\nabla \alpha\right) b_i + \alpha (\nabla b_i)\right) \cdot \left(k \nabla b_j\right), \\
  &= -\int T_j \left(\left(-\frac{\left(\nabla \rho\right) C_p + \rho \left(\nabla C_p\right)}{\left(\rho C_p\right)^2}\right) b_i + \alpha (\nabla b_i)\right) \cdot \left(k \nabla b_j\right).
\end{align}
\\$$  

If we assume that $$\rho$$ and $$C_p$$ only depend on the temperature, we get:

$$\\
\begin{align}
-\int T_j \nabla \left(\alpha b_i\right) \cdot \left(k \nabla b_j\right) 
  &= -\int T_j \left(\left(-\frac{\frac{\partial \rho}{\partial T} C_p\left(\nabla T\right) + \rho \frac{\partial C_p}{\partial T}\left(\nabla T\right)}{\left(\rho C_p\right)^2}\right) b_i + \alpha (\nabla b_i)\right) \cdot \left(k \nabla b_j\right), \\
  &= -\int T_j \left(\left(-\frac{\left(\frac{\partial \rho}{\partial T} C_p + \rho \frac{\partial C_p}{\partial T}\right)\left(\nabla T\right)}{\left(\rho C_p\right)^2}\right) b_i + \alpha (\nabla b_i)\right) \cdot \left(k \nabla b_j\right), \\
\end{align}
\\
$$
