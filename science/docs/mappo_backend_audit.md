# Phase 2C-1 MAPPO backend audit

Audit date: 17 September 2026, before Phase 2C-1.5 migration.

## Classification

`OLD MAPPO BACKEND = NUMPY`

`OLD GRADIENT MECHANISM = NUMPY_MANUAL_BACKPROP`

The accepted Phase 2C-1 implementation is not an external automatic-differentiation implementation and does not use finite differences. It evaluates a one-hidden-layer actor and critic with NumPy and applies explicitly derived analytical derivatives in ordinary Python/NumPy code.

## Existing actor path

`src/rl/mappo/actor.py` stores `W1`, `b1`, `Wmu`, `bmu`, and `log_std` as NumPy arrays. Forward evaluation uses matrix multiplication and `numpy.tanh`. `SharedActor.log_prob_gradients` explicitly calculates the diagonal-Gaussian derivatives with respect to the mean and log standard deviation and then manually propagates them through the mean head and hidden tanh layer. No tensor graph or `.backward()` call exists.

## Existing critic path

`src/rl/mappo/critic.py` stores `W1`, `b1`, `Wout`, and `bout` as NumPy arrays. `CentralizedCritic.loss_and_gradients` manually forms the value error, output-head derivatives, and hidden-layer derivatives. It is analytical manual backpropagation, not numerical perturbation.

## Existing optimizer and PPO path

`src/rl/mappo/trainer.py` implements Adam directly with NumPy first/second moments and parameter-array mutation. The PPO clipped-surrogate derivative with respect to nominal-action log probability is assembled manually, then passed to the actor's analytical gradient function. Critic gradients are likewise supplied explicitly to the custom Adam optimizer.

## Existing checkpoint representation

Phase 2C-1 checkpoints are Python pickle payloads containing copied NumPy parameter arrays, custom Adam moment dictionaries, NumPy normalizer state, Python and NumPy RNG states, steps, configuration hash, and provenance flags. PyTorch state was recorded as unavailable.

## Why the earlier gradient status passed

The Phase 2C-1 report's actor/critic gradient status meant that the repository's explicit analytical NumPy gradients were finite, nonzero where expected, shared across trucks, clipped, and able to update parameters in the DEBUG PPO smoke. It did **not** mean a production autodiff framework was installed. Phase 2C-2 correctly found that PyTorch, JAX, TensorFlow, and Autograd were unavailable. Phase 2C-1.5 therefore replaces the canonical training backend with one unified PyTorch actor, critic, optimizer, and PPO loss before any DiffQP work.

If migration becomes possible, the old NumPy equations may remain only as a migration oracle; they must not remain a selectable training backend. In the present host, migration stopped before source changes because Windows application control prevents importing the official PyTorch binaries.
