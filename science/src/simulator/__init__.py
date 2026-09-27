"""Validated Phase-2A continuous heavy-vehicle simulator interfaces."""

from .auxiliary_brake import AuxiliaryForceInterface, SyntheticDebugAuxiliaryEnvelope
from .fade import FadeModel, TabulatedFadeCurve, synthetic_debug_fade_curve
from .gearbox import GearState, GearboxParameters, advance_dwell_timer, can_shift, shift_gear
from .integrators import (
    SimulationResult,
    integrate_reference,
    integrate_zero_order_hold,
    integrate_zero_order_hold_reference,
    rk4_step,
)
from .truck_dynamics import (
    EnvironmentInput,
    SimulatorConfig,
    StateDerivative,
    VehicleCommand,
    VehicleParameters,
    VehicleState,
    load_synthetic_debug_config,
    vehicle_rhs,
)

__all__ = [
    "AuxiliaryForceInterface",
    "SyntheticDebugAuxiliaryEnvelope",
    "FadeModel",
    "TabulatedFadeCurve",
    "synthetic_debug_fade_curve",
    "GearState",
    "GearboxParameters",
    "advance_dwell_timer",
    "can_shift",
    "shift_gear",
    "SimulationResult",
    "integrate_reference",
    "integrate_zero_order_hold",
    "integrate_zero_order_hold_reference",
    "rk4_step",
    "EnvironmentInput",
    "SimulatorConfig",
    "StateDerivative",
    "VehicleCommand",
    "VehicleParameters",
    "VehicleState",
    "load_synthetic_debug_config",
    "vehicle_rhs",
]
