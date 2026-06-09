# Research and mathematical references

These references cover the mathematical and machine-learning components used by the implementation.

## Articulated vehicles and trailer systems

- Rouchon, Fliess, Lévine, and Martin, **“Flatness, Motion Planning and Trailer Systems”**. This develops flatness-based motion planning for standard and general trailer systems and uses Frénet formulae for path geometry.
- Altafini, **“Some Properties of the General N-Trailer”**. This studies the nonholonomic structure of the general N-trailer model.
- Ljungqvist, Axehill, and Helmersson, **“Path Following Control for a Reversing General 2-Trailer System”**, arXiv:1605.04393. This treats stabilisation and path following for reversing trailer systems.
- Pasillas-Lépine and Respondek, **“Nilpotentization of the Kinematics of the N-Trailer System at Singular Points and Motion Planning Through the Singular Locus”**, arXiv:math/0004125. This addresses singular configurations and constructive normal forms for N-trailer kinematics.
- Singh, Jayakumar, and Rizzoni, **“An Iterative Algorithm to Symbolically Derive Generalized N-Trailer Vehicle Kinematics”**, arXiv:2504.00315. This derives control-oriented yaw-plane kinematic models for generalized articulated vehicles.

## Sampling-based model predictive control

- Williams, Aldrich, and Theodorou, **“Model Predictive Path Integral Control using Covariance Variable Importance Sampling”**, arXiv:1509.01149. This is the MPPI-style sampling control backbone used by the expert controller.
- Kim, Park, Kwak, Bae, and Lee, **“Smooth Model Predictive Path Integral Control Without Smoothing”**, IEEE Robotics and Automation Letters, 2022. This motivates smooth sampling-based action sequences for nonlinear systems.

## Gaussian kernels, random features, and high-dimensional feature maps

- Rahimi and Recht, **“Random Features for Large-Scale Kernel Machines”**, NeurIPS 2007. This justifies approximating Gaussian/RBF kernels with finite randomized feature maps.
- Jayasumana, Hartley, Salzmann, Li, and Harandi, **“Kernel Methods on Riemannian Manifolds with Gaussian RBF Kernels”**, arXiv:1412.0265. This supports using Gaussian-style kernels on manifold-valued data under metric constraints.
- Hutchinson, Terenin, Borovitskiy, Takao, Teh, and Gretton, **“Vector-valued Gaussian Processes on Riemannian Manifolds”**, NeurIPS 2021. This gives a coordinate-free framework for Gaussian vector fields on manifolds.
- Mallasto and Feragen, **“Wrapped Gaussian Process Regression on Riemannian Manifolds”**, CVPR 2018. This supports tangent-space wrapping/log-map intuitions for probabilistic regression on curved spaces.

## Multivariate Gaussian diagnostics

- Henze and Zirkler, **“A Class of Invariant Consistent Tests for Multivariate Normality”**, Communications in Statistics - Theory and Methods, 1990. This motivates optional diagnostics for whether learned latent rollouts retain approximately Gaussian structure.
