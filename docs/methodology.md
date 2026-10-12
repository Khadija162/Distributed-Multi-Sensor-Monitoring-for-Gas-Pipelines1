# Methodology and scientific boundaries

Research basis: Shaheen, Chawla and Salvo Rossi, *Trust-Enhanced Distributed Kalman Filtering for Sensor Fault Diagnosis in Sensor Networks*, IEEE TSIPN vol 11 (2025), DOI **10.1109/TSIPN.2025.3606167**. This project is an independent educational implementation, not the authors' code.

Paper method and implementation mapping:
1. Eqs (1)-(4): local 4D state model and Kalman time/covariance prediction in `tedkf/model.py`. A is the paper's thermal transition matrix; H=I, Q=0.05I; R defaults to 0.1I, a demo configuration.
2. Eq (5): individual innovation y_i = z_i - H x_i prior.
3. Eqs (6)-(8): IDB trusted innovation selection using min mutual reachability to the previous trusted innovation cluster. Gamma defaults to 5; reference core distances exclude self; at the first time all nodes initialize the trust cluster.
4. Eq (9): adaptive threshold tau = mean(detection scores)+lambda. Lambda defaults to 0.35, a demonstration setting, not a claimed optimized paper value.
5. Eq (10): mean innovation over trusted nodes (no state or covariance messages exchanged).
6. Eq (11): measurement update uses shared trusted innovation with local Kalman gain. Joseph covariance update is a numerical-stability modification.

**Deviations and caveats:** The simulation uses invented noise and injection scales and only the paper's illustrative thermal transition model, not gas fluid mechanics. The density reference is one time step of trusted innovations; mutual-reachability nearest-neighbor interpretation is explicit. This implementation does not reproduce the authors' empirical accuracy tables or ROC curves. It has no communication latency, authenticated networking, signal constraints, device calibration, or live telemetry.

The paper's first step trusts all nodes, so a corrupted initialization compromises the reference. Mean+lambda threshold can also be overly permissive when many faults dominate the current mean. If all nodes are rejected, this implementation skips the innovation update and keeps the previous reference instead of inventing reliable measurements. A validated gas pipeline application requires a calibrated nonlinear hydraulic state-space model, operational topology, secure distributed communications, strong testing, fail-safe human review and independent sensor validation.
