# Symbol audit

SI units are mandatory in code and logs. `DATA REQUIRED` means that a physical value must be supplied from manufacturer, dynamometer, coast-down, CAN-bus, map, or held-out identification data before a vehicle-level claim is permitted.

| Symbol | Definition / first equation | Unit | Role | Code variable | Source / calibration | Hard guarantee? | Defined before use? |
|---|---|---:|---|---|---|---:|---:|
| $p_i$ | longitudinal position, (1a) | m | state | external simulator | localization | yes | yes |
| $v_i$ | longitudinal speed, (1a) | m/s | state | `State.speed_mps` | wheel speed/GNSS | yes | yes |
| $a_i$ | longitudinal acceleration | m/s2 | derived state | `State.acceleration_mps2` | IMU/model consistency | yes | yes |
| $b_i$ | realized friction-brake force, (1b) | N | state | `State.friction_force_N` | brake-pressure/force identification | yes | yes |
| $r_i$ | realized auxiliary-brake force, (1b) | N | state | `State.auxiliary_force_N` | retarder/engine map | yes | yes |
| $T_i$ | effective brake temperature, (1c) | K | state | `State.temperature_K` | dynamometer/thermocouple | yes | yes |
| $u_i^f,u_i^a$ | friction and auxiliary force commands | N | control | command tuple | actuator command scaling | yes | yes |
| $F_i^d$ | propulsive/driveline force | N | input | folded into `nonbraking_force_N` | powertrain model; DATA REQUIRED | yes | yes |
| $F_i^r$ | total nonbraking longitudinal force | N | derived input | `nonbraking_force_N` | force balance | yes | yes |
| $\dot F_i^r$ | nonbraking-force rate | N/s | uncertain input | `nonbraking_force_rate_Nps` | interval bound; DATA REQUIRED | yes | yes |
| $\rho$ | air density | kg/m3 | parameter | external model | weather/source data | yes | yes |
| $w_i$ | unmodeled longitudinal-force residual | N | disturbance | interval component | held-out residual bound | yes | yes |
| $q_i^w$ | thermal residual heat flow | W | disturbance | `heat_residual_W` | held-out thermal residual bound | yes | yes |
| $\dot q_i^w$ | thermal residual rate | W/s | disturbance | `heat_residual_rate_Wps` | derivative bound; DATA REQUIRED | yes | yes |
| $j_i$ | continuous-time jerk, Eq. jerk | m/s3 | derived | `continuous_jerk` | dynamics | soft only | yes |
| $e_i^d$ | spacing error for string-stability metric | m | metric | result log | prescribed spacing policy | no | yes |
| $\epsilon_g$ | nonzero metric regularizer | m | metric parameter | config TODO | registered constant | no | yes |
| $g_i$ | discrete gear/mode | 1 | mode | external mode id | CAN/powertrain model | yes | yes |
| $\bar r_i(v,g)$ | realized auxiliary-force envelope | N | parameter map | `realized_limit_N` | retarder/engine map; DATA REQUIRED | yes | yes |
| $\bar u_i^a(v,g)$ | auxiliary command envelope | N | parameter map | `auxiliary_envelope_limit_N` | power/gear map; DATA REQUIRED | yes | yes |
| $a_i^S$ | follower conservative braking magnitude | m/s2 | safety parameter | `follower_safe_decel_mps2` | validated lower bound | yes | yes |
| $a_{i-1}^E$ | predecessor emergency braking magnitude | m/s2 | safety parameter | `predecessor_emergency_decel_mps2` | declared reachable set | yes | yes |
| $v_\epsilon$ | low-speed switching threshold | m/s | design parameter | `low_speed_threshold_mps` | numerical/validation choice | yes | yes |
| $\Delta_T$ | one-step thermal buffer | K | design parameter | `temperature_buffer_K` | validation margin | yes | yes |
| $\Delta t$ | zero-order-hold sample time | s | design parameter | `sample_time_s` | controller clock | yes | yes |
| $\zeta_i$ | uncertain variables entering uncontrolled barrier terms | mixed | uncertainty vector | interval inputs | Sec. IV-E declaration | yes | yes |
| $\mathcal B_i(\hat\zeta_i)$ | Cartesian interval uncertainty box | mixed | uncertainty set | `Interval` objects | sensing/comms/model bounds | yes | yes |
| $\gamma_i^k$ | one-sided robust margin | barrier derivative unit | derived bound | `directional_margin` | interval arithmetic | yes | yes |
| $L_i^k$ | intersample residual-rate bound | barrier derivative unit/s | derived bound | `rate_bound` | interval/Jacobian bound; DATA REQUIRED | yes | yes |
| $\bar e_{\rm int,i}^k$ | integration/reconstruction error bound | barrier derivative unit | derived bound | `integration_error` | integrator validation | yes | yes |
| $\mu_i^k$ | sampled-data tightening | barrier derivative unit | derived bound | `sampled_data_margin` | $L_i^k\Delta t+\bar e_{\rm int}$ | yes | yes |
| $D_i^c$ | collision braking demand | m/s2 | derived | `collision["D_c"]` | collision HOCBF | yes | yes |
| $L_i^f,U_i^f$ | effective friction command limits | N | derived limits | `limits["L_f/U_f"]` | intersection of all hard rows | yes | yes |
| $L_i^a,U_i^a$ | effective auxiliary command limits | N | derived limits | `limits["L_a/U_a"]` | envelope, rate, auxiliary CBF | yes | yes |
| $M_i$ | signed physical conflict reserve | m/s2 | certificate | `conflict_reserve` | exact interval-plus-collision geometry | yes | yes |
| $M_{\rm ref}$ | desired predicted reserve | m/s2 | learning weight target | actor config TODO | validation/tuning only | no | yes |
| $s_{im}$ | generic polytope row scale | row unit | diagnostic parameter | not used in hard certificate | fixed physical scale | no | yes |
| $\sigma_i$ | optional Chebyshev diagnostic | 1 | diagnostic | not differentiated | general-polytope monitor | no | yes |
| $T_{\rm soft}$ | soft thermal target | K | learning target | actor config TODO | validation/tuning | no | yes |
| $h_{\rm ref}^F$ | desired fade reserve | N | learning target | actor config TODO | validation/tuning | no | yes |
| $M_{\rm trig},M_{\rm rec}$ | backup trigger/recovery reserve | m/s2 | supervisor parameters | config TODO | validation | yes | yes |
| $N_{\rm rec}$ | consecutive recovery samples | sample | supervisor parameter | config TODO | registered integer | yes | yes |
| $W_i,\rho_j,\epsilon$ | QP tracking/slack/regularization weights | scaled | optimizer parameters | projection config TODO | solver tuning | no | yes |
| $\lambda_I,\lambda_M,\lambda_T,\lambda_F,\beta$ | actor auxiliary/entropy weights | scaled | learning parameters | training config TODO | validation split only | no | yes |
| $H_{\rm pred}$ | differentiable prediction horizon | sample | learning parameter | training config TODO | registered before training | no | yes |
| $H_{\rm QP}$ | positive-definite QP Hessian | scaled | optimizer matrix | projection implementation | constructed from weights | no | yes |
| $g^{T0}$ | undivided low-speed state-only thermal reserve | K | hard certificate input | `low_speed_temperature_row.g_T0_K` | calibrated thermal model and error bound | yes | yes |
| $s_{T0}$ | low-speed thermal normalizer | K | fixed certificate scale | `CertificateScales.low_speed_temperature_K` | registered physical scale | yes | yes |
| $q^{T0}$ | active normalized low-speed thermal reserve, otherwise $+\infty$ | 1 | certificate component | `CompleteCertificateResult.normalized_low_speed_thermal` | computed | yes | yes |

Notation collision repaired: recovery duration is $N_{\rm rec}$, prediction horizon is $H_{\rm pred}$, and the QP Hessian is $H_{\rm QP}$.
