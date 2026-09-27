"""Complete Phase 2C-1 checkpoint and reproducibility state handling."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import pickle
import platform
import random
from typing import Any
import numpy as np

from .actor import SharedActor
from .critic import CentralizedCritic
from .trainer import AdamOptimizer


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def configuration_hash(configuration: Any) -> str:
    encoded = json.dumps(_jsonable(configuration), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import torch  # type: ignore
    except ImportError:
        versions["torch"] = "unavailable"
        versions["torch_cuda"] = "unavailable"
    else:
        versions["torch"] = str(torch.__version__)
        versions["torch_cuda"] = str(torch.version.cuda or "unavailable")
    return versions


def initialize_reproducibility(
    python_seed: int,
    numpy_seed: int,
    environment_seed: int,
    network_seed: int,
    scenario_seed: int,
) -> tuple[np.random.Generator, dict[str, object]]:
    random.seed(python_seed)
    np.random.seed(numpy_seed)
    generator = np.random.default_rng(network_seed)
    torch_record: dict[str, object]
    try:
        import torch  # type: ignore
    except ImportError:
        torch_record = {"torch_cpu_seed": None, "torch_cuda_seed": None, "torch_deterministic": False, "reason": "torch unavailable; NumPy implementation active"}
    else:
        torch.manual_seed(network_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(network_seed)
        torch.use_deterministic_algorithms(True)
        torch_record = {
            "torch_cpu_seed": network_seed,
            "torch_cuda_seed": network_seed if torch.cuda.is_available() else None,
            "torch_deterministic": True,
        }
    record: dict[str, object] = {
        "python_seed": python_seed,
        "numpy_seed": numpy_seed,
        "environment_seed": environment_seed,
        "network_seed": network_seed,
        "scenario_seed": scenario_seed,
        **torch_record,
        "library_versions": library_versions(),
        "data_provenance": "synthetic_debug",
        "paper_eligible": False,
        "training_mode": "debug",
    }
    return generator, record


def save_checkpoint(
    path: Path,
    actor: SharedActor,
    critic: CentralizedCritic,
    actor_optimizer: AdamOptimizer,
    critic_optimizer: AdamOptimizer,
    training_step: int,
    environment_step: int,
    rng: np.random.Generator,
    configuration: Any,
    reproducibility: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "training_step": int(training_step),
        "environment_step": int(environment_step),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "generator_state": rng.bit_generator.state,
        "config_hash": configuration_hash(configuration),
        "reproducibility": reproducibility,
        "library_versions": library_versions(),
        "data_provenance": "synthetic_debug",
        "paper_eligible": False,
        "training_mode": "debug",
        "diffqp_backward_active": False,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def load_checkpoint(
    path: Path,
    actor: SharedActor,
    critic: CentralizedCritic,
    actor_optimizer: AdamOptimizer,
    critic_optimizer: AdamOptimizer,
    rng: np.random.Generator,
    expected_configuration: Any | None = None,
) -> dict[str, object]:
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("paper_eligible") is not False or payload.get("training_mode") != "debug":
        raise ValueError("checkpoint is not an allowed Phase 2C-1 DEBUG artifact")
    if payload.get("diffqp_backward_active") is not False:
        raise ValueError("Phase 2C-1 checkpoint unexpectedly enables DiffQP backward")
    if expected_configuration is not None and payload["config_hash"] != configuration_hash(expected_configuration):
        raise ValueError("checkpoint configuration hash mismatch")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor_optimizer.load_state_dict(payload["actor_optimizer"])
    critic_optimizer.load_state_dict(payload["critic_optimizer"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    rng.bit_generator.state = payload["generator_state"]
    return payload
