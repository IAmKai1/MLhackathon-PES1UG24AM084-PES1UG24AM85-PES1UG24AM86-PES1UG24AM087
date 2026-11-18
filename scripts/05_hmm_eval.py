"""HMM/Viterbi route reconstruction: decoder tuning and classifier-only vs classifier+HMM.

1. Decoder parameters (beta, emission_weight) are grid-searched on TRAINING trips using
   out-of-fold classifier probabilities from 04_bayes_opt.py (no test data involved).
2. The tuned final model + tuned decoder are evaluated on the 10 student-testing trips
   through the same pipeline the API uses, and compared with classifier-only predictions.
3. Folium maps of GPS, network, reference and predicted routes are saved per test trip.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_trip  # noqa: E402
from pathsnap.hmm import HMMConfig, viterbi_decode  # noqa: E402
from pathsnap.maps import trip_map  # noqa: E402
from pathsnap.metrics import route_continuity  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.pipeline import run_trip  # noqa: E402
from pathsnap.plotstyle import SERIES, plt, save  # noqa: E402

FEAT_DIR = ROOT / "results" / "features"
OUT = ROOT / "results" / "hmm"
MAPS = ROOT / "results" / "maps"
ART = ROOT / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
MAPS.mkdir(parents=True, exist_ok=True)
N_TUNE_TRIPS = 16
# Only beta * emission_weight affects the Viterbi argmax (scaling both rescales every
# path score), so the grid spans that product from 0.5 to 80.
BETAS = [1.0, 1.5, 2.0, 3.0, 6.0, 10.0, 20.0, 40.0]
WEIGHTS = [0.5, 1.0, 2.0]


def decode_from_frame(g, trip, net, cfg):
    g = g.sort_values(["obs", "p"], ascending=[True, False])
    grp = g.groupby("obs", sort=True)
    obs_ids = np.fromiter(grp.groups.keys(), dtype=np.int64)
    arcs = [x.arc.to_numpy() for _, x in grp]
    ts = [x.proj_t.to_numpy() for _, x in grp]
    ps = [x.p.to_numpy() for _, x in grp]
    path, breaks = viterbi_decode(obs_ids, arcs, ts, ps, trip.xy, trip.t, net, cfg)
    clf = np.array([a[0] for a in arcs])
    lab = grp.label_arc.first().to_numpy()
    ok = grp.label_ok.first().to_numpy()
    return obs_ids, clf, path, lab, ok, breaks


def tune_decoder():
    train = pd.read_pickle(FEAT_DIR / "train.pkl")
    train["p"] = np.load(ROOT / "results" / "bayes_opt" / "train_oof_proba_tuned.npy")
    # Trips with reliable labels, spread across sizes; capped at 4000 observations for speed
    info = train.groupby("trip").agg(n=("obs", "max"), ok=("label_ok", "mean"))
    pool = info[(info.ok > 0.95) & (info.n < 4000)].index.tolist()
    rng = np.random.default_rng(42)
    tune_trips = sorted(rng.choice(pool, size=min(N_TUNE_TRIPS, len(pool)), replace=False).tolist())
    cache = {tid: (lambda t: (t, RoadNetwork(t.node_xy, t.arcs)))(load_trip(ROOT / "data" / "train" / tid))
             for tid in tune_trips}
    rows = []
    for beta in BETAS:
        for w in WEIGHTS:
            cfg = HMMConfig(beta=beta, emission_weight=w)
            hits = tot = 0
            clf_hits = 0
            conts = []
            t0 = time.time()
            for tid in tune_trips:
                trip, net = cache[tid]
                obs_ids, clf, path, lab, ok, _ = decode_from_frame(train[train.trip == tid], trip, net, cfg)
                hits += int((path[ok] == lab[ok]).sum())
                clf_hits += int((clf[ok] == lab[ok]).sum())
                tot += int(ok.sum())
                conts.append(route_continuity(path, net))
            rows.append({"beta": beta, "emission_weight": w, "hmm_top1_acc": hits / tot,
                         "classifier_top1_acc": clf_hits / tot, "hmm_continuity": float(np.mean(conts)),
                         "seconds": time.time() - t0})
            print(rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "decoder_tuning_train_oof.csv", index=False)
    best = grid.loc[grid.hmm_top1_acc.idxmax()]
    cfg = HMMConfig(beta=float(best.beta), emission_weight=float(best.emission_weight))
    json.dump({**cfg.to_dict(), "tuned_on": f"{len(tune_trips)} training trips (out-of-fold probabilities)",
               "tune_trips": tune_trips}, open(ART / "hmm_config.json", "w"), indent=2)
    return cfg, grid


def main():
    cfg, grid = tune_decoder()
    model = joblib.load(ART / "final_model.joblib")
    rows = []
    for trip_dir in sorted((ROOT / "data" / "student_test").iterdir()):
        r = run_trip(trip_dir, model, cfg)
        sc, sh = r["score_classifier"], r["score_hmm"]
        rows.append({
            "trip": trip_dir.name, "n_obs": r["n_observations"], "chain_breaks": r["chain_breaks"],
            "clf_obs_acc": sc["observation_accuracy"], "hmm_obs_acc": sh["observation_accuracy"],
            "clf_continuity": r["continuity_classifier"], "hmm_continuity": r["continuity_hmm"],
            "clf_route_f1": sc["route_set_f1"], "hmm_route_f1": sh["route_set_f1"],
            "trusted_obs": sh["trusted_observations"],
            "time_features_s": r["time_features_s"], "time_decode_s": r["time_decode_s"],
        })
        print(rows[-1], flush=True)
        trip_map(r["trip"], r["net"], MAPS / f"student_test_{trip_dir.name}.html", predicted_arcs=r["hmm_arcs"],
                 title=f"Student-test trip {trip_dir.name}: HMM accuracy {sh['observation_accuracy']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "student_test_classifier_vs_hmm.csv", index=False)
    w = res.trusted_obs / res.trusted_obs.sum()
    overall = {
        "classifier_obs_acc_weighted": float((res.clf_obs_acc * w).sum()),
        "hmm_obs_acc_weighted": float((res.hmm_obs_acc * w).sum()),
        "classifier_continuity_mean": float(res.clf_continuity.mean()),
        "hmm_continuity_mean": float(res.hmm_continuity.mean()),
        "classifier_route_f1_mean": float(res.clf_route_f1.mean()),
        "hmm_route_f1_mean": float(res.hmm_route_f1.mean()),
        "decoder": cfg.to_dict(),
    }
    json.dump(overall, open(OUT / "student_test_summary.json", "w"), indent=2)
    print(json.dumps(overall, indent=2))

    x = np.arange(len(res))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (c, h, title) in zip(axes, [("clf_obs_acc", "hmm_obs_acc", "Observation accuracy"),
                                        ("clf_continuity", "hmm_continuity", "Route continuity")]):
        ax.bar(x - 0.2, res[c], 0.38, color=SERIES[0], label="Classifier only")
        ax.bar(x + 0.2, res[h], 0.38, color=SERIES[1], label="Classifier + HMM")
        ax.set_xticks(x, res.trip.str[-2:], fontsize=7)
        ax.set(title=f"{title} per student-test trip", xlabel="trip (last 2 digits)",
               ylim=(min(0.5, res[[c, h]].min().min() - 0.05), 1.01))
        ax.legend(loc="lower right")
    save(fig, OUT / "classifier_vs_hmm.png")

    piv = grid.pivot(index="beta", columns="emission_weight", values="hmm_top1_acc")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for i, wgt in enumerate(piv.columns):
        ax.plot(piv.index, piv[wgt], marker="o", markersize=5, color=SERIES[i], label=f"emission weight {wgt}")
    ax.axhline(grid.classifier_top1_acc.iloc[0], color="#52514e", linestyle="--", linewidth=1)
    ax.text(piv.index.max(), grid.classifier_top1_acc.iloc[0], "classifier only", ha="right", va="bottom",
            fontsize=7, color="#52514e")
    ax.set_xscale("log")
    ax.set(title="Decoder tuning on training trips (out-of-fold)", xlabel="beta (m)", ylabel="Top-1 accuracy")
    ax.legend()
    save(fig, OUT / "decoder_tuning.png")


if __name__ == "__main__":
    main()
