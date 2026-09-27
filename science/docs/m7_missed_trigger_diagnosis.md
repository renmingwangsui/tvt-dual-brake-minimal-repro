# M7 Nominal-delay Missed-trigger Diagnosis

All 20 missed records used valid messages aged exactly one 0.1 s control sample. Counts by receiving vehicle were {'1': 9, '2': 6, '3': 5}. Critical-vehicle metadata changed or disagreed in 10 records. The centralized reserve lay -0.00178618 to -0.000306359 relative to the fixed 0.08 threshold, while the local delayed reserve remained 0.000451668 to 0.0319271 above it.

The causes are sampled packet-consumption timing, causal recursive multi-hop aggregation lag, and bottleneck-vehicle changes near the supervisor threshold. There was no packet loss, staleness, horizon mismatch, parameter mismatch, or verified implementation defect. No communication parameter or threshold was changed, and M7 was not rerun. The distributed-bottleneck claim remains **PARTIALLY_SUPPORTED**.
