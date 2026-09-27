"""DEBUG-only Phase 2C-2 training smoke for matched QP gradient modes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from envs.heavy_platoon_env import HeavyPlatoonEnv
from envs.scenario import build_synthetic_debug_scenario
from safety.diffqp import (
    CanonicalSafetyQP,
    QPDebugRecord,
    QPGradientMode,
    solve_differentiable_qp,
    write_debug_qp_log,
)
from safety_core import Row

from .actor import local_observation_vector
from .checkpoint import initialize_reproducibility
from .critic import centralized_state_vector
from .reward import DebugReward, DebugRewardWeights
from .torch_backend import (
    TorchCentralizedCritic,
    TorchMAPPOTrainer,
    TorchSharedActor,
    collect_torch_rollout,
    save_torch_checkpoint,
    torch_ppo_losses,
)
from .trainer import PPOConfig, PPOUpdateMetrics


@dataclass(frozen=True)
class Phase2C2DebugTrainingResult:
    mode: str
    actor_checksum_before: str
    actor_checksum_after: str
    ppo_metrics: PPOUpdateMetrics
    qp_actions: tuple[tuple[float, float], ...]
    qp_intervention_max: float
    qp_mediated_gradient_norm: float
    qp_actor_parameter_gradient_norm: float
    actor_gradient_norm: float
    safety_stack_verified: bool
    checkpoint_path: str
    checkpoint_config_hash: str
    fallback_count: int
    fallback_reasons: tuple[str, ...]
    data_provenance: str = "DEBUG"
    paper_eligible: bool = False
    training_mode: str = "debug"


def registered_intervention_qp() -> CanonicalSafetyQP:
    """Stable one-row intervention with the accepted row/solver semantics."""
    rows = (
        Row("friction_lower", -1.0, 0.0, 0.0, "N"),
        Row("friction_upper", 1.0, 0.0, 120_000.0, "N"),
        Row("auxiliary_lower", 0.0, -1.0, 0.0, "N"),
        Row("auxiliary_upper", 0.0, 1.0, 80_000.0, "N"),
        Row("collision_hocbf", -1.0, -1.0, -190_000.0, "m/s^2"),
    )
    return CanonicalSafetyQP(rows)


def run_phase2c2_debug_training_smoke(
    mode: QPGradientMode,
    checkpoint_path: Path,
    seed: int = 20260923,
) -> Phase2C2DebugTrainingResult:
    """Run one matched N=3 DEBUG job; this is not a performance experiment."""
    rng, reproducibility = initialize_reproducibility(seed, seed + 1, seed + 2, seed + 3, seed + 4)
    scenario = build_synthetic_debug_scenario(3, "flat_steady")
    frozen_safety_parameters = repr(scenario.parameters)
    env = HeavyPlatoonEnv(scenario)
    observation_dim = local_observation_vector(env.current_local_observations()[0]).size
    state_dim = centralized_state_vector(env.centralized_training_state()).size
    upper = np.asarray([
        min(item.friction_force_limit_N for item in scenario.parameters),
        min(item.auxiliary_force_limit_N for item in scenario.parameters),
    ])
    actor = TorchSharedActor(observation_dim, 16, np.zeros(2), upper, rng)
    critic = TorchCentralizedCritic(state_dim, 16, 3, rng)
    config = PPOConfig(epochs=1, minibatch_size=6)
    trainer = TorchMAPPOTrainer(actor, critic, config)
    reward = DebugReward(
        DebugRewardWeights(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01), 18.0, 65.0
    )

    rollout, summary = collect_torch_rollout(env, actor, critic, reward, 2)
    ppo_metrics = trainer.update(rollout, rng)
    if summary.nominal_executed_difference_count <= 0:
        raise AssertionError("registered DEBUG rollout did not exercise the safety filter")
    safety_verified = (
        "all_commands_verified" in env.execution_trace
        and all(record.final_residuals_valid for record in env.logs)
    )
    if not safety_verified:
        raise AssertionError("complete safety stack was not verified during Phase 2C-2 smoke")

    observations = np.stack([local_observation_vector(item) for item in env.current_local_observations()])
    mean, _ = actor(observations)
    u_rl = actor.distribution.deterministic(mean)
    qp = registered_intervention_qp()
    qp_results = [solve_differentiable_qp(u_rl[index], qp, mode) for index in range(3)]
    qp_actions = torch.stack([result.action for result in qp_results])
    interventions = torch.linalg.vector_norm(qp_actions - u_rl, dim=1)
    if not bool(torch.all(interventions > 0.0).item()):
        raise AssertionError("registered QP-mediated training case must be nontrivial")

    # Ordinary PPO likelihood remains on stored u_RL.  The additional DEBUG
    # validation term isolates the QP-mediated path without redefining PPO.
    frozen_batch = rollout.flattened()
    ordinary_ppo_actor_loss = torch_ppo_losses(actor, critic, frozen_batch, config).actor_loss
    qp_validation_loss = 0.05 * torch.mean((qp_actions[:, 0] / 120_000.0).square())
    mediated_gradient = torch.autograd.grad(
        qp_validation_loss, u_rl, retain_graph=True, allow_unused=False
    )[0]
    mediated_gradient_norm = float(torch.linalg.vector_norm(mediated_gradient).item())
    parameter_gradients = torch.autograd.grad(
        qp_validation_loss,
        tuple(actor.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    qp_actor_parameter_gradient_norm = float(np.sqrt(sum(
        float(torch.sum(gradient.square()).item())
        for gradient in parameter_gradients
        if gradient is not None
    )))
    if mode is QPGradientMode.DIFFQP and not (np.isfinite(mediated_gradient_norm) and mediated_gradient_norm > 0.0):
        raise AssertionError("DiffQP smoke did not produce a finite nonzero QP-mediated gradient")
    if mode is QPGradientMode.DIFFQP and not (
        np.isfinite(qp_actor_parameter_gradient_norm) and qp_actor_parameter_gradient_norm > 0.0
    ):
        raise AssertionError("DiffQP QP-mediated gradient did not reach actor parameters")
    if mode is QPGradientMode.STOP_GRADIENT and mediated_gradient_norm != 0.0:
        raise AssertionError("StopGradient smoke leaked a QP-mediated gradient")
    if mode is QPGradientMode.STOP_GRADIENT and qp_actor_parameter_gradient_norm != 0.0:
        raise AssertionError("StopGradient smoke leaked QP-mediated actor gradients")

    actor_before = actor.checksum()
    trainer.actor_optimizer.zero_grad(set_to_none=True)
    (ordinary_ppo_actor_loss + qp_validation_loss).backward()
    trainer._validate_gradients(actor, "actor")
    actor_gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), config.maximum_gradient_norm
    )
    trainer.actor_optimizer.step()
    actor_after = actor.checksum()
    if actor_before == actor_after:
        raise AssertionError("Phase 2C-2 DEBUG actor update changed no parameters")
    actor_gradient_norm = float(actor_gradient_norm_tensor.item())
    for result in qp_results:
        result.record.actor_gradient_norm = actor_gradient_norm

    if repr(scenario.parameters) != frozen_safety_parameters:
        raise AssertionError("Phase 2C-2 training modified safety/physical parameters")
    configuration = {"scenario": "flat_steady", "N": 3, "horizon": 2, "ppo": config}
    payload = save_torch_checkpoint(
        checkpoint_path,
        actor,
        critic,
        trainer,
        environment_step=6,
        rng=rng,
        configuration=configuration,
        reproducibility=reproducibility,
        qp_gradient_mode=mode.value,
    )
    write_debug_qp_log(Path(checkpoint_path).with_suffix(".qp_debug.json"), [
        result.record for result in qp_results
    ])
    fallback_reasons = tuple(
        result.record.fallback_reason
        for result in qp_results
        if result.record.fallback_reason not in {"NONE", "STOP_GRADIENT_MODE"}
    )
    return Phase2C2DebugTrainingResult(
        mode=mode.value,
        actor_checksum_before=actor_before,
        actor_checksum_after=actor_after,
        ppo_metrics=ppo_metrics,
        qp_actions=tuple(tuple(map(float, row)) for row in qp_actions.detach().cpu().numpy()),
        qp_intervention_max=float(torch.max(interventions).detach().item()),
        qp_mediated_gradient_norm=mediated_gradient_norm,
        qp_actor_parameter_gradient_norm=qp_actor_parameter_gradient_norm,
        actor_gradient_norm=actor_gradient_norm,
        safety_stack_verified=safety_verified,
        checkpoint_path=str(checkpoint_path),
        checkpoint_config_hash=str(payload["config_hash"]),
        fallback_count=len(fallback_reasons),
        fallback_reasons=fallback_reasons,
    )
