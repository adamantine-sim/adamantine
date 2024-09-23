---
layout: default
parent: Home
title: Data Assimilation
nav_order: 3
usemathjax: true
---

# Data Assimilation
The goal of data assimilation is to optimally combine a numerical simulation
with observations. In *adamantine*, experimental data from infra-red cameras 
and thermo-couples can be used to improve the simulation. We perform the data 
assimilation using the stochastic Ensemble Kalman filter method (EnKF). For an 
in-depth discussion about this algorithm, we recommend 
[Data Assimilation](https://epubs.siam.org/doi/abs/10.1137/1.9781611974546) by 
Mark Asch, Marc Bocquet, and Maëlle Nodet. 

EnKF combines simulations and observations as follows:

$$
x_i^a = x_i^f + K [y-H z_i^f],
$$

with:

$$
K = P^f H^T (H P^f H^T +R)^{-1},
$$

where $$x_i^a$$ is the $$i^{th}$$ updated simulation of the ensemble, $$x_i^f$$ 
is the $$i^{th}$$ simulation of the ensemble, $$K$$ is the Kalman gain, $$y$$ is
the observations (the experimental data), $$H$$ is the observation matrix that maps
the simulation to the observation, $$P$$ is the simulation error covariance, and
$$R$$ is the obversation error covariance. $$R$$ depends on the instruments. $$P^f$$ is
given by:

$$
P^f = \frac{1}{m-1} \sum_{i=1}^M (x_i^f - \bar{x}^f) (x_i^f - \bar{x}^f)^T,
$$

with:

$$
\bar{x}^f = \frac{1}{m} \sum_{i=1}^m x_i^f.
$$

$$m$$ is the number of ensemble simulations.

The [Bare plate L]({{site.baseurl}}/doc/examples/bare_plate_l) example
demonstrates *adamantine* data assimilation capabilities.

As of version 1.0 of *adamantine* data assimilation is restricted to thermal
simulations.
