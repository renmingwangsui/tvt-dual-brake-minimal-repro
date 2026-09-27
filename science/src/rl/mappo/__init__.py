"""PyTorch MAPPO core with explicit matched DiffQP/StopGradient support.

The accepted NumPy/manual implementation remains explicitly exported as the
numerical migration reference. DiffQP is opt-in; the default safety-forward
controller and PPO likelihood semantics remain unchanged.
"""

from .actor import RunningNormalizer, SharedActor as NumpySharedActor, local_observation_vector
from .buffer import RolloutBatch, RolloutBuffer
from .critic import CentralizedCritic as NumpyCentralizedCritic, centralized_state_vector
from .distributions import SquashedGaussian as NumpySquashedGaussian, SquashedGaussianSample as NumpySquashedGaussianSample
from .gae import generalized_advantage_estimation as numpy_generalized_advantage_estimation
from .checkpoint import (
    configuration_hash, initialize_reproducibility,
    load_checkpoint as load_numpy_checkpoint, save_checkpoint as save_numpy_checkpoint,
)
from .evaluation import EvaluationResult as NumpyEvaluationResult, evaluate_deterministic as evaluate_numpy_deterministic
from .reward import DebugReward, DebugRewardWeights, RewardBreakdown
from .rollout import RolloutSummary as NumpyRolloutSummary, collect_rollout as collect_numpy_rollout
from .trainer import (
    AdamOptimizer as NumpyAdamOptimizer, DebugTrainingSmokeResult as NumpyDebugTrainingSmokeResult,
    MAPPOTrainer as NumpyMAPPOTrainer, PPOConfig, PPOUpdateMetrics,
    clipped_surrogate, ppo_probability_ratio, run_debug_training_smoke as run_numpy_debug_training_smoke,
)
from .torch_backend import (
    TorchCentralizedCritic, TorchDebugTrainingSmokeResult, TorchEvaluationResult,
    TorchMAPPOTrainer, TorchRolloutSummary, TorchSharedActor, TorchSquashedGaussian,
    TorchSquashedGaussianSample, collect_torch_rollout, evaluate_torch_deterministic,
    load_torch_checkpoint, run_torch_debug_training_smoke, save_torch_checkpoint,
    torch_clipped_surrogate, torch_generalized_advantage_estimation,
    torch_ppo_losses, torch_probability_ratio,
)
from .diffqp_training import (
    Phase2C2DebugTrainingResult,
    registered_intervention_qp,
    run_phase2c2_debug_training_smoke,
)

# Package-level runtime defaults are now the genuine autograd backend.
SharedActor = TorchSharedActor
CentralizedCritic = TorchCentralizedCritic
SquashedGaussian = TorchSquashedGaussian
SquashedGaussianSample = TorchSquashedGaussianSample
MAPPOTrainer = TorchMAPPOTrainer
collect_rollout = collect_torch_rollout
evaluate_deterministic = evaluate_torch_deterministic
save_checkpoint = save_torch_checkpoint
load_checkpoint = load_torch_checkpoint
generalized_advantage_estimation = torch_generalized_advantage_estimation
EvaluationResult = TorchEvaluationResult
RolloutSummary = TorchRolloutSummary
DebugTrainingSmokeResult = TorchDebugTrainingSmokeResult
run_debug_training_smoke = run_torch_debug_training_smoke
AdamOptimizer = NumpyAdamOptimizer

DIFFQP_BACKWARD_ACTIVE = False

__all__ = [
    "RunningNormalizer", "SharedActor", "NumpySharedActor", "local_observation_vector",
    "RolloutBatch", "RolloutBuffer", "CentralizedCritic", "centralized_state_vector",
    "SquashedGaussian", "SquashedGaussianSample", "generalized_advantage_estimation",
    "configuration_hash", "initialize_reproducibility", "load_checkpoint", "save_checkpoint",
    "EvaluationResult", "evaluate_deterministic", "DebugReward", "DebugRewardWeights",
    "RewardBreakdown", "RolloutSummary", "collect_rollout", "MAPPOTrainer",
    "DebugTrainingSmokeResult", "PPOConfig", "PPOUpdateMetrics", "clipped_surrogate",
    "ppo_probability_ratio", "run_debug_training_smoke",
    "NumpyCentralizedCritic", "NumpySquashedGaussian", "NumpySquashedGaussianSample",
    "numpy_generalized_advantage_estimation", "load_numpy_checkpoint", "save_numpy_checkpoint",
    "NumpyEvaluationResult", "evaluate_numpy_deterministic", "NumpyRolloutSummary",
    "collect_numpy_rollout", "NumpyAdamOptimizer", "NumpyDebugTrainingSmokeResult",
    "NumpyMAPPOTrainer", "run_numpy_debug_training_smoke", "AdamOptimizer",
    "TorchSharedActor", "TorchCentralizedCritic",
    "TorchSquashedGaussian", "TorchSquashedGaussianSample", "TorchMAPPOTrainer",
    "collect_torch_rollout", "evaluate_torch_deterministic", "save_torch_checkpoint",
    "load_torch_checkpoint", "run_torch_debug_training_smoke", "torch_clipped_surrogate",
    "torch_generalized_advantage_estimation", "torch_ppo_losses", "torch_probability_ratio",
    "Phase2C2DebugTrainingResult", "registered_intervention_qp",
    "run_phase2c2_debug_training_smoke",
    "DIFFQP_BACKWARD_ACTIVE",
]
