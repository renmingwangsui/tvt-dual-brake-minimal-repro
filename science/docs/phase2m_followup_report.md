# Phase 2M Targeted Follow-up

Status: **PASS WITH RETAINED M8 LIMITATION**.

The frozen physical configuration remained `d10d13eb82490bcfb2151f7db2114e38d8896810255b3138d17540cddabd5104`. No physical parameter, uncertainty bound, communication calibration, supervisor threshold, MAPPO component, or DiffQP component was changed. M1--M10 were not rerun.

## M8 branch diagnosis

The physical state and each thermal row vary smoothly around 0.5 m/s. At the hot state, `g_T0=-2.8399562779 K`; the active certificate changes from the one-step ZOH row to the thermal HOCBF row between 0.5 and 0.500001 m/s, producing a rho jump of 406.09776454. The change in `c_T` is only 1.8455837656e-12 K/N and the counterfactual HOCBF limit changes by only -0.0835116614326 N. Thus the jump is not caused by physical dynamics or floating-point tolerance. It is the declared hard switch between non-equivalent sufficient conditions.

No repair was applied. A transition blend or overlap cannot be certified without extending the registered ZOH error-validity domain or changing the hard-set definition. Continuity at `v_epsilon` is therefore not claimed.

## Targeted predictive warning

The scenario grid was saved and hashed before execution. Across 81 scenarios: 10 true early warnings, 32 conservative warnings without a unique boundary, 39 simultaneous warnings, and 0 missed warnings were observed. 4 true-warning scenarios satisfied both the pre-registered adjacency rule and finite numerical margins. The targeted predictive-warning claim is **SUPPORTED**, limited to this frozen operational family; no universal time-to-conflict theorem is implied.

## M7 and M9

M7's 20 nominal-delay misses are expected delayed-information effects, not an implementation defect. M9 remains unchanged: max G2=4.21649, max Ginf=5.16703; the pre-registered numerical string-stability claim is **NOT_SUPPORTED**.
