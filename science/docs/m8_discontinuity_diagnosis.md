# M8 Low-speed Discontinuity Diagnosis

- Root cause: hard switch between non-equivalent low-speed one-step ZOH and high-speed thermal-HOCBF sufficient conditions.
- Genuine physical/model discontinuity: no.
- Low-speed ZOH condition contributes: yes (`g_T0=-2.8399562779 K`).
- Hard implementation branch contributes: yes.
- Normalization changes magnitude but not the feasibility sign: yes.
- Numerical tolerance cause: no.
- Hot rho jump at 0.5 to 0.500001 m/s: 406.09776454.
- Cold feasibility sign flip: False.
- Hot feasibility sign flip: True.
- Repair required under the frozen declared model: no.
- Repair status: `NOT_APPLIED_UNDER_FROZEN_VALIDITY_DOMAINS`.

Extending/blending the ZOH row above its registered domain or inventing an overlap width would change certificate assumptions without a registered error bound. The piecewise semantics are retained and continuity is explicitly disclaimed.
