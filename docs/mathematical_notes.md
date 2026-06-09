# Mathematical notes

## N-trailer kinematics

The simulator represents an articulated chain as

\[
z = (x_0, y_0, \theta_0, \alpha_1, \ldots, \alpha_N),
\quad \alpha_i = \theta_{i-1} - \theta_i.
\]

The tractor obeys a bicycle-style yaw rate

\[
\dot{x}_0 = v \cos \theta_0,\qquad
\dot{y}_0 = v \sin \theta_0,\qquad
\dot{\theta}_0 = \frac{v}{L_0}\tan \delta.
\]

For trailer `i`, longitudinal velocity transfers through the articulation angle:

\[
v_i = v_{i-1}\cos \alpha_i,\qquad
\dot{\theta}_i = \frac{v_i}{L_i}\sin\alpha_i,
\qquad
\dot{\alpha}_i = \dot{\theta}_{i-1} - \dot{\theta}_i.
\]

Reverse motion uses `v < 0`. Jackknife risk is penalised by a barrier on `|alpha_i|`.

## Path-local sphere lift

For a nearest path frame `(p(s), psi(s), kappa(s))`, define

\[
q_p = \frac{(\cos\psi, \sin\psi, \tanh(\kappa/c_\kappa))}
{\|(\cos\psi, \sin\psi, \tanh(\kappa/c_\kappa))\|_2},
\]

and, for tractor heading `theta` and signed cross-track error `e`,

\[
q_v = \frac{(\cos\theta, \sin\theta, \tanh(e/c_e))}
{\|(\cos\theta, \sin\theta, \tanh(e/c_e))\|_2}.
\]

The tangent feature is the logarithmic map on the sphere:

\[
\xi = \log_{q_p}(q_v)
= \frac{\omega}{\sin \omega}\left(q_v - \cos\omega\,q_p\right),
\quad \omega = \arccos(q_p^\top q_v).
\]

Projecting `xi` onto an orthonormal basis of the tangent plane gives two smooth arc coordinates. These coordinates are stable around small heading/cross-track errors and remain bounded near large deviations.

## Gaussian combinatorial feature field

A Gaussian RBF kernel can be approximated by random Fourier features

\[
\phi(x) = \sqrt{\frac{2}{M}}\cos(Wx + b),
\quad W_{ij}\sim \mathcal{N}(0, \sigma^{-2}),
\quad b_j\sim U(0, 2\pi).
\]

Sparse subset interactions are appended as

\[
\eta_j(x) = \exp\left(-\frac{\|S_jx-a_j\|_2^2}{2\ell_j^2}\right)
\prod_{k\in S_j} \operatorname{sgn}(r_{jk}) x_k,
\]

where `S_j` selects a small subset of coordinates. The subset terms couple tangent coordinates, curvature, and articulation angles without requiring a dense polynomial expansion.
