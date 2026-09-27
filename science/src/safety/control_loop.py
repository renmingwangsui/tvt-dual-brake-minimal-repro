"""Executable two-pass sampled control loop with enforced causal ordering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .predictive_reserve import final_control_requires_reverification


@dataclass(frozen=True)
class TwoPassCallbacks:
    measure_and_align: Callable[[], object]
    build_margins_and_intervals: Callable[[object], object]
    sample_nominal_actor: Callable[[object, object], Sequence[float]]
    solve_normal_qp: Callable[[Sequence[float], object], Sequence[float]]
    predict_from_provisional: Callable[[object, Sequence[float]], object]
    aggregate_bottleneck: Callable[[object], object]
    select_final_command: Callable[[Sequence[float], object, object], tuple[str, Sequence[float]]]
    fast_verify_final_prediction: Callable[[object, Sequence[float]], object]
    validate_hard_rows: Callable[[object, Sequence[float]], None]
    apply_command: Callable[[Sequence[float]], None]
    log_cycle: Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class TwoPassCycleResult:
    mode: str
    nominal_command: tuple[float, float]
    provisional_command: tuple[float, float]
    final_command: tuple[float, float]
    provisional_prediction: object
    final_prediction: object
    prediction_reverified: bool


def execute_two_pass_control_cycle(callbacks: TwoPassCallbacks) -> TwoPassCycleResult:
    """Execute the ten manuscript steps in a single non-reorderable routine."""
    state = callbacks.measure_and_align()                                      # 1
    hard_context = callbacks.build_margins_and_intervals(state)                # 2
    nominal = tuple(map(float, callbacks.sample_nominal_actor(state, hard_context)))  # 3
    provisional = tuple(map(float, callbacks.solve_normal_qp(nominal, hard_context))) # 4
    prediction0 = callbacks.predict_from_provisional(state, provisional)       # 5
    aggregate = callbacks.aggregate_bottleneck(prediction0)                    # 6
    mode, selected = callbacks.select_final_command(provisional, prediction0, aggregate) # 7
    final = tuple(map(float, selected))
    reverify = final_control_requires_reverification(provisional, final)
    prediction_final = callbacks.fast_verify_final_prediction(state, final) if reverify else prediction0 # 8
    callbacks.validate_hard_rows(state, final)                                 # 9
    callbacks.apply_command(final)
    record = {
        "mode": mode,
        "nominal_command": nominal,
        "provisional_command": provisional,
        "final_command": final,
        "provisional_prediction": prediction0,
        "final_prediction": prediction_final,
        "prediction_reverified": reverify,
    }
    callbacks.log_cycle(record)                                                # 10
    return TwoPassCycleResult(
        mode, nominal, provisional, final, prediction0, prediction_final, reverify
    )
