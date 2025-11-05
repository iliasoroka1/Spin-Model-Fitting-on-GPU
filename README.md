# Spin Model Fitting on GPU

GPU-accelerated implementation of Ising model sampling and maximum entropy fitting using JAX and THRML.

## Overview

This package provides tools for maximum entropy parameter fitting to match target statistical properties (mean and correlations).

## Background

### The Ising Model

The Ising model describes a system of $N$ interacting binary spins $s_i = \pm 1$. The energy of a configuration $\mathbf{s} = (s_1, \ldots, s_N)$ is:

 $$E(\mathbf{s}) = -\sum_{i<j} J_{ij} s_i s_j - \sum_i h_i s_i$$

where:
- $J_{ij}$ are pairwise couplings (interactions between random variables)
- $h_i$ are external fields acting on each spin (exogenous variables, biases)
- The first sum runs over all pairs of spins $(i,j)$ with $i < j$

### Boltzmann Distribution

At inverse temperature $\beta = 1/T$, the probability of observing configuration $\mathbf{s}$ follows the Boltzmann distribution:

$$P(\mathbf{s}) = \frac{1}{Z} \exp(-\beta E(\mathbf{s}))$$

where $Z = \sum_{\mathbf{s}} \exp(-\beta E(\mathbf{s}))$ is the partition function (normalization constant).

### Observable Statistics

From the Boltzmann distribution, we can compute expectation values:

**Magnetizations** (single-spin averages):
$$\langle s_i \rangle = \sum_{\mathbf{s}} s_i P(\mathbf{s})$$

**Correlations** (two-spin averages):
$$\langle s_i s_j \rangle = \sum_{\mathbf{s}} s_i s_j P(\mathbf{s})$$

### Gibbs Sampling

Since exact summation over all $2^N$ configurations is intractable for large $N$, we use **Gibbs sampling** to draw samples from $P(\mathbf{s})$. The conditional probability for spin $i$ given all others is:

$$P(s_i | \mathbf{s}_{-i}) = \frac{\exp(\beta s_i h_i^{\text{eff}})}{\exp(\beta h_i^{\text{eff}}) + \exp(-\beta h_i^{\text{eff}})}$$

where the effective field is:
$$h_i^{\text{eff}} = h_i + \sum_{j \neq i} J_{ij} s_j$$

### Maximum Entropy Principle

Given target statistics $\langle s_i \rangle^{\text{target}}$ and $\langle s_i s_j \rangle^{\text{target}}$, we want to find parameters $(J, h)$ such that the model's statistics match the targets.

The **maximum entropy principle** states that among all distributions matching the constraints, the one with maximum entropy is the Boltzmann distribution. This leads to the Lagrangian optimization problem:

$$\mathcal{L}(J, h) = \sum_i \left(\langle s_i \rangle^{\text{model}} - \langle s_i \rangle^{\text{target}}\right)^2 + \sum_{i<j} \left(\langle s_i s_j \rangle^{\text{model}} - \langle s_i s_j \rangle^{\text{target}}\right)^2$$

### Gradient Descent with Momentum

The gradients of the log-likelihood with respect to parameters are simple:

$$\frac{\partial \log P}{\partial h_i} = \langle s_i \rangle^{\text{target}} - \langle s_i \rangle^{\text{model}}$$

$$\frac{\partial \log P}{\partial J_{ij}} = \langle s_i s_j \rangle^{\text{target}} - \langle s_i s_j \rangle^{\text{model}}$$

We use momentum-based gradient ascent:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta \log P$$
$$\theta_{t+1} = \theta_t + v_t$$

where $\eta$ is the learning rate and $\gamma$ is the momentum.

## Usage

```python
import numpy as np
from model import MaxEntBoltzmann

# suppose we observed these statistics from an unknown system
N = 5
target_magnetizations = np.array([0.2, -0.3, 0.1, 0.4, -0.2])

target_correlations = np.array([
    [1.0,  0.5,  0.1,  0.0, -0.1],
    [0.5,  1.0,  0.6,  0.2,  0.0],
    [0.1,  0.6,  1.0,  0.4,  0.1],
    [0.0,  0.2,  0.4,  1.0,  0.3],
    [-0.1, 0.0,  0.1,  0.3,  1.0]
])

fitter = MaxEntBoltzmannTHRML(
    n=N,
    target_s=target_magnetizations,
    target_ss=target_correlations,
    beta=1.0
)

J_fit, h_fit = fitter.fit(
    n_iterations=50,
    learning_rate=0.1,
    momentum=0.9,
    n_chains=1000,      
    n_steps=200,
    n_burn_in=100,
    adaptive_lr=True,   
    verbose=True
)

```
