"""Fixed-N centralized critic with explicit per-agent outputs."""
from __future__ import annotations

from hashlib import sha256
import numpy as np

from envs.observations import CentralizedTrainingState


def centralized_state_vector(state: CentralizedTrainingState) -> np.ndarray:
    values: list[float] = list(state.leader_state)
    for vehicle in state.controlled_states:
        values.extend((
            vehicle.position_m,
            vehicle.speed_mps,
            vehicle.friction_force_N,
            vehicle.auxiliary_force_N,
            vehicle.temperature_K,
        ))
    values.extend(float(valid) for valid in state.communication_validity)
    return np.asarray(values, dtype=np.float64)


class CentralizedCritic:
    def __init__(self, state_dim: int, hidden_dim: int, controlled_truck_count: int, rng: np.random.Generator) -> None:
        if controlled_truck_count < 1:
            raise ValueError("critic requires a fixed positive N")
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.controlled_truck_count = controlled_truck_count
        self.parameters: dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, 1.0 / np.sqrt(state_dim), (state_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "Wout": rng.normal(0.0, 0.05, (hidden_dim, controlled_truck_count)),
            "bout": np.zeros(controlled_truck_count),
        }

    def forward(self, states: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(states, dtype=np.float64))
        if x.shape[1] != self.state_dim:
            raise ValueError("critic state dimension mismatch; fixed N cannot be silently padded")
        hidden = np.tanh(x @ self.parameters["W1"] + self.parameters["b1"])
        return hidden @ self.parameters["Wout"] + self.parameters["bout"]

    def loss_and_gradients(
        self, states: np.ndarray, agent_indices: np.ndarray, targets: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
        x = np.atleast_2d(np.asarray(states, dtype=np.float64))
        hidden = np.tanh(x @ self.parameters["W1"] + self.parameters["b1"])
        all_values = hidden @ self.parameters["Wout"] + self.parameters["bout"]
        indices = np.asarray(agent_indices, dtype=np.int64)
        selected = all_values[np.arange(len(x)), indices]
        error = selected - np.asarray(targets, dtype=np.float64)
        if not np.all(np.isfinite(error)):
            raise FloatingPointError("non-finite critic error")
        loss = 0.5 * float(np.mean(error**2))
        dout = np.zeros_like(all_values)
        dout[np.arange(len(x)), indices] = error / len(x)
        dhidden = dout @ self.parameters["Wout"].T
        dpre = dhidden * (1.0 - hidden**2)
        gradients = {
            "Wout": hidden.T @ dout,
            "bout": np.sum(dout, axis=0),
            "W1": x.T @ dpre,
            "b1": np.sum(dpre, axis=0),
        }
        return loss, gradients, selected

    def state_dict(self) -> dict[str, object]:
        return {
            "parameters": {name: value.copy() for name, value in self.parameters.items()},
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "controlled_truck_count": self.controlled_truck_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = (self.state_dim, self.hidden_dim, self.controlled_truck_count)
        actual = (int(state["state_dim"]), int(state["hidden_dim"]), int(state["controlled_truck_count"]))
        if actual != expected:
            raise ValueError("critic checkpoint architecture/N mismatch")
        for name in self.parameters:
            self.parameters[name][...] = np.asarray(state["parameters"][name], dtype=np.float64)

    def checksum(self) -> str:
        digest = sha256()
        for name in sorted(self.parameters):
            digest.update(name.encode())
            digest.update(self.parameters[name].tobytes())
        return digest.hexdigest()
