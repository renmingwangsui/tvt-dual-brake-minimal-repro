# Phase 2C-1 MAPPO design

## Scope and evidence boundary

This implementation is a software-validation MAPPO core around the accepted Phase 2B environment. Every run is marked `data_provenance = synthetic_debug`, `paper_eligible = false`, and `training_mode = debug`. The values in the DEBUG reward configuration are not production reward weights. The safety controller participates only in the forward transition; `DIFFQP_BACKWARD_ACTIVE` is `False` and no KKT or implicit-QP gradient is implemented.

The implementation uses an explicit NumPy multilayer perceptron and checkpointable Adam optimizer because PyTorch is unavailable in the validated runtime. Reproducibility metadata records that condition rather than pretending PyTorch or CUDA seeds were applied.

## Decentralized actor and information boundary

One parameter-shared actor is used for all controlled trucks. Its 11-dimensional input is the canonical `LocalActorObservation`: own speed, realized friction and auxiliary forces, own brake temperature, local gap, relative speed, previous two commands, gear identifier, message age, and received safety intent. It receives no fleet state, fleet minimum, future predecessor state, realized disturbance, other vehicle state, or critic-only diagnostic.

The actor is a one-hidden-layer tanh MLP with a two-dimensional Gaussian head and a learned, clipped diagonal `log_std`. A single actor optimizer consumes the flattened samples from every vehicle, so heterogeneous vehicle physics do not create per-truck actors or optimizers.

Running observation statistics are updated only from the current training environment at a rollout boundary and are held fixed throughout that rollout and its PPO update. Evaluation neither updates nor replaces them.

## Centralized critic and registered N

The critic consumes the canonical centralized training state: leader position/speed/acceleration, all controlled physical states, and communication-validity flags. It outputs one value per controlled truck. Each training run registers a fixed `N`; a dimension mismatch is rejected instead of silently padding vehicles without a mask.

## Squashed Gaussian and physical actions

For each nominal action dimension,

\[
z\sim\mathcal N(\mu,\sigma^2),\qquad y=\tanh z,\qquad
u=c+s y,
\]

where \(c=(u_{\max}+u_{\min})/2\) and \(s=(u_{\max}-u_{\min})/2\). For an interior physical action, the implementation evaluates

\[
\log \pi(u\mid o)=\sum_j\left[
\log\mathcal N(z_j;\mu_j,\sigma_j^2)
-\log s_j-\log(1-\tanh^2 z_j)
\right],
\]

with \(z=\operatorname{atanh}((u-c)/s)\). Thus the PPO ratio is not an unsquashed Gaussian ratio and includes both the tanh and affine Jacobians.

## Nominal, provisional, and executed actions

The buffer preserves three independent arrays:

- `nominal_actions`: actor sample \(u_{RL}\), used for old/new log probability and the PPO ratio;
- `provisional_actions`: normal hard-QP output \(u_0^\star\);
- `executed_actions`: final verified command \(u_{final}\), used by the physical simulator.

The transition is therefore actor \(\rightarrow u_{RL}\rightarrow\) existing hard QP/prediction/supervisor/possible second action/re-verification \(\rightarrow u_{final}\rightarrow s_{k+1}\). No rollout shortcut calls the simulator with \(u_{RL}\), and no likelihood is evaluated for \(u_{final}\).

## Buffer and reward

Every step stores local observations, centralized state, all three action concepts, old nominal-action log probability, reward, terminal and truncation flags, value and next value, supervisor mode, intervention norm, \(\rho\), \(\rho_H^{cert}\), critical vehicle/component, and message validity. Diagnostic fields do not enter the PPO objective.

The modular DEBUG reward exposes speed tracking, spacing tracking, control effort, friction use, auxiliary use, intervention, and optional safety-penalty terms. Its dataclass rejects paper eligibility. Production weights remain `CONFIG_REQUIRED`/`DATA_REQUIRED`.

## GAE and PPO

For a true terminal, value bootstrap is zero. For a pure time-limit truncation, \(V_{t+1}\) remains in the one-step delta, while advantage recursion stops at the episode boundary:

\[
\delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t,
\quad
A_t=\delta_t+\gamma\lambda(1-\max(d_t,q_t))A_{t+1},
\]

where \(d_t\) is terminal and \(q_t\) is truncation.

PPO uses \(r_t(\theta)=\exp[\log\pi_\theta(u_{RL}|o)-\log\pi_{old}(u_{RL}|o)]\), the clipped surrogate, a centralized value loss, and a sampled entropy term. Actor and critic gradients are independently norm-clipped and checked for NaN/Inf. Metrics report policy loss, value loss, entropy, approximate KL, clip fraction, and pre-clipping actor/critic gradient norms.

## Checkpoint and evaluation

Checkpoints contain actor and critic parameters, both Adam states, normalization statistics, training and environment steps, Python/NumPy/generator RNG states, configuration hash, seeds, library versions, and DEBUG/DiffQP flags. Load checks the configuration hash and restores RNG state.

Deterministic evaluation uses the Gaussian mean followed by tanh and affine scaling. It performs no exploration sample, optimizer step, or normalization update. Evaluation still sends every nominal command through the complete Phase 2B safety forward path.
