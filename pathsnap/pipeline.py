"""End-to-end prediction for one trip: candidates -> classifier -> HMM/Viterbi.

Used by both the evaluation scripts and the API, so the demonstrated system and
the reported experiments run exactly the same code. Reference labels are never
passed to the model: features are built with `with_labels=False`, and labels are
only computed afterwards (server-side) for scoring.
"""
import time

import numpy as np

from .data import load_trip
from .features import FEATURES, build_trip_frame
from .hmm import HMMConfig, viterbi_decode
from .labels import align_route
from .metrics import route_continuity, route_set_f1

LABEL_MAX_DIST = 30.0


def predict_frame(df, trip, net, model, cfg: HMMConfig, use_hmm=True):
    """Return (obs_ids, classifier_arcs, hmm_arcs, n_breaks) for the trip's candidate frame."""
    df = df.copy()
    df["p"] = model.predict_proba(df[FEATURES].to_numpy(np.float32))[:, 1]
    df = df.sort_values(["obs", "p"], ascending=[True, False])
    groups = df.groupby("obs", sort=True)
    obs_ids = np.fromiter(groups.groups.keys(), dtype=np.int64)
    arcs = [g.arc.to_numpy() for _, g in groups]
    ts = [g.proj_t.to_numpy() for _, g in groups]
    ps = [g.p.to_numpy() for _, g in groups]
    clf_arcs = np.array([a[0] for a in arcs])            # highest probability first
    if not use_hmm:
        return obs_ids, clf_arcs, clf_arcs, 0
    hmm_arcs, n_breaks = viterbi_decode(obs_ids, arcs, ts, ps, trip.xy, trip.t, net, cfg)
    return obs_ids, clf_arcs, hmm_arcs, n_breaks


def fill_all_observations(n, obs_ids, arcs):
    """Predicted arc for every observation; points without candidates take the previous prediction."""
    full = np.full(n, -1, dtype=np.int64)
    full[obs_ids] = arcs
    last = -1
    for i in range(n):
        if full[i] < 0:
            full[i] = last
        last = full[i]
    first = np.flatnonzero(full >= 0)
    if len(first):
        full[: first[0]] = full[first[0]]
    return full


def score_trip(trip, net, pred_full):
    """Server-side scoring against the reference route (labels never reach the model or UI)."""
    labels, d = align_route(trip, net)
    ok = d <= LABEL_MAX_DIST
    return {
        "observation_accuracy": float(np.mean(pred_full[ok] == labels[ok])) if ok.any() else None,
        "trusted_observations": int(ok.sum()),
        "route_set_f1": route_set_f1(pred_full[pred_full >= 0], trip.route),
    }


def run_trip(trip_dir, model, cfg: HMMConfig, score=True):
    t0 = time.time()
    trip = load_trip(trip_dir, with_route=True)
    route = trip.route
    trip.route = None                                   # hide the reference from feature building
    df, cand, _, net = build_trip_frame(trip, with_labels=False)
    t_feat = time.time()
    obs_ids, clf_arcs, hmm_arcs, n_breaks = predict_frame(df, trip, net, model, cfg)
    t_pred = time.time()
    n = len(trip.xy)
    clf_full = fill_all_observations(n, obs_ids, clf_arcs)
    hmm_full = fill_all_observations(n, obs_ids, hmm_arcs)
    out = {
        "trip": trip, "net": net, "classifier_arcs": clf_full, "hmm_arcs": hmm_full,
        "n_observations": n, "observations_with_candidates": int(len(obs_ids)), "chain_breaks": int(n_breaks),
        "continuity_classifier": route_continuity(clf_full, net),
        "continuity_hmm": route_continuity(hmm_full, net),
        "time_features_s": t_feat - t0, "time_decode_s": t_pred - t_feat,
    }
    if score and route is not None:
        trip.route = route
        out["score_classifier"] = score_trip(trip, net, clf_full)
        out["score_hmm"] = score_trip(trip, net, hmm_full)
    return out
