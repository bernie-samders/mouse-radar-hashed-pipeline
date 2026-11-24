"""variants.py -- arm pll: named tracker recipes for estimate.py (every logged exploration variant is in
results_full.jsonl with its cfg in `notes`; these are the recipes estimate.py can regenerate blind).

Constants not listed take pf.BASE / pll.BASE defaults (see RESULT.md for provenance):
  sig_s 0.11, L_beats 5, sig_w 0.06  transfer (dev-mouse ECG IBI statistics, dimensionless; fit.py 508 / 507d2)
  N 1024, dec 5 (50 Hz), lag_s 1.0, outdec 2, ckf 1, acap 0.5, loho, mu_em   a_priori
  qscale 2.0                         fit_split (FIT least-squares scale alpha* ~ 1 at qscale 2 vs 1.4 at 1)
  em_iter 2, zgate 4, ks 1..5        fit_split (FIT ve1 +0.025 vs k1-only +0.016; LOHO essential: no-LOHO -0.005)
"""
K1 = dict(engine="pf", ks=[1], em_iter=0, qscale=2.0)
H = dict(engine="pf", em_iter=2, qscale=2.0, zgate=4)
G1 = dict(engine="pf", ks=[1], em_iter=0, qscale=2.0, half1=0.372, guard=0.10)
GH = dict(engine="pf", em_iter=2, qscale=2.0, zgate=4, half1=0.372, guard=0.10)
FZ = dict(engine="pf", ks=[1], em_iter=0, qscale=2.0, half1=0.372, guard=0.10, bidir=1, nrep=4)
HB = dict(engine="pf", em_iter=2, qscale=2.0, zgate=4, half1=0.372, guard=0.10, bandgain=2, acap=0.2)
FH = dict(engine="pf", em_iter=2, qscale=2.0, zgate=16, half1=0.372, guard=0.10, bandgain=2, acap=0.2, bidir=1, nrep=2)
ALL = {
    "pfh_g45_z16_bd_r2": FH,
    "pf1_g45_bd_r4": FZ,
    "pfh_g45_bg2_ac02": HB,
    "pf1_g45": G1,
    "pfh_g45": GH,
    "pf_k1_q2": K1,
    "pfh": H,
    "ekf_k1": dict(engine="ekf", ks=[1], em_iter=0),
}
FROZEN_NAME = "pfh_g45_z16_bd_r2"
