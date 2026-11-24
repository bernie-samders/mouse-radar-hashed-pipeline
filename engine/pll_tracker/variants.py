"""variants.py -- every beam configuration evaluated (evalb.py rounds r1..r9, in order), plus the FROZEN choice.

Selection (declared in evalb.py before r1 was run): INFO = sum_k k^2 max(0, median-over-FIT-periods per-beat SNR_k)
of the ECG-locked harmonic amplitude of the beam (headroom statistic on OUR weights), FIT split only.
r1..r9: best single recipe k1sel5_lamk3 (FIT INFO 1.63). r10: per-harmonic cross-fitted contrast choice between
k1sel5_lamk3 and the frozen-picks eigenbeam -> FIT INFO 2.03 (pooled 3.7 vs 2.55) -> FROZEN = choose_k1sel5L3_pickeig
(51 configurations; chosen on FIT; TEST reported only).
"""
ROUNDS = {
    "r1": {"pickeig": {"sel": "picks", "ref": "none"}, "pickgev_ring": {"sel": "picks", "ref": "ring"},
           "all_ring": {"sel": "all", "ref": "ring"}, "all_broad": {"sel": "all", "ref": "broad"},
           "top20_ring": {"sel": "topM", "M": 20, "ref": "ring"}, "top40_ring": {"sel": "topM", "M": 40, "ref": "ring"}},
    "r2": {"all_ring_cap": {"ringcap": 1}, "all_ring_din08": {"din": 0.077}, "all_ring_din15": {"din": 0.14},
           "all_ring_lam03": {"lamf": 0.3}, "all_ring_lam3": {"lamf": 3.0}, "all_ring_dd1": {"dd": 1},
           "all_ring_dd2": {"dd": 2}, "top40_ring_dd1": {"sel": "topM", "M": 40, "dd": 1}},
    "r3": {"ring_k1sel10": {"hk": "k1sel", "Mk": 10}, "ring_k1sel20": {"hk": "k1sel", "Mk": 20},
           "ring_k1sel40": {"hk": "k1sel", "Mk": 40}, "ring_couple": {"hk": "couple"},
           "ring_couple_lam3": {"hk": "couple", "lamf": 3.0}},
    "r4": {"trk4_all": {"track_s": 4.0}, "trk4_k1sel20": {"track_s": 4.0, "hk": "k1sel", "Mk": 20},
           "trk2_k1sel20": {"track_s": 2.0, "hk": "k1sel", "Mk": 20},
           "trk8_k1sel20": {"track_s": 8.0, "hk": "k1sel", "Mk": 20},
           "trk4_k1sel20_din06": {"track_s": 4.0, "hk": "k1sel", "Mk": 20, "din": 0.058}},
    "r5": {"k1sel10_din06": {"hk": "k1sel", "Mk": 10, "din": 0.058}, "k1sel20_din06": {"hk": "k1sel", "Mk": 20, "din": 0.058},
           "k1sel20_mix1": {"hk": "k1sel", "Mk": 20, "selmix": 1.0}, "k1sel20_lamk3": {"hk": "k1sel", "Mk": 20, "lamk": 3.0},
           "k1sel20_lamk03": {"hk": "k1sel", "Mk": 20, "lamk": 0.3}, "k1sel5": {"hk": "k1sel", "Mk": 5}},
    "r6": {"prior_ring": {"hk": "prior"}, "prior_in": {"hk": "prior", "prior_R": "in"},
           "prior_ring_lam3": {"hk": "prior", "lamk": 3.0}, "prior_ring_lam03": {"hk": "prior", "lamk": 0.3},
           "prior_all5_ring": {"hk": "prior", "prior_k1": 1}},
    "r7": {"comb_all_k1to5": {"comb": 1},
           "comb_all_k2to5": {"comb": 1, "comb_k0": 2, "hk": "k1sel", "Mk": 10, "din": 0.058},
           "comb_all_k2to5_k2w": {"comb": 1, "comb_k0": 2, "comb_w": "k2"},
           "comb_sel40_k2to5": {"comb": 1, "comb_k0": 2, "comb_sel": 40}, "comb_sel20_k1to5": {"comb": 1, "comb_sel": 20}},
    "r8": {"k1sel5_lamk3": {"hk": "k1sel", "Mk": 5, "lamk": 3.0},
           "k1sel10_din06_lamk3": {"hk": "k1sel", "Mk": 10, "din": 0.058, "lamk": 3.0},
           "gev1_pickeig": {"hk": "pickeig"}, "k1sel5_din06": {"hk": "k1sel", "Mk": 5, "din": 0.058},
           "k1sel3": {"hk": "k1sel", "Mk": 3}},
    "r9": {"k1sel5_lamk10": {"hk": "k1sel", "Mk": 5, "lamk": 10.0}, "k1sel8_lamk3": {"hk": "k1sel", "Mk": 8, "lamk": 3.0},
           "k1sel3_lamk3": {"hk": "k1sel", "Mk": 3, "lamk": 3.0}},
    "r10": {"choose_k1sel5L3_pickeig": {"choose": [{"hk": "k1sel", "Mk": 5, "lamk": 3.0}, {"hk": "pickeig"}]},
            "choose_k1sel5L3_pickeig_allring": {"choose": [{"hk": "k1sel", "Mk": 5, "lamk": 3.0}, {"hk": "pickeig"}, {}]},
            "choose_k1sel5L3_k1sel20_pickeig": {"choose": [{"hk": "k1sel", "Mk": 5, "lamk": 3.0}, {"hk": "k1sel", "Mk": 20},
                                                           {"hk": "pickeig"}]}},
}
ALL = {n: c for r in ROUNDS.values() for n, c in r.items()}
FROZEN_NAME = "choose_k1sel5L3_pickeig"
FROZEN = dict(ALL[FROZEN_NAME])
