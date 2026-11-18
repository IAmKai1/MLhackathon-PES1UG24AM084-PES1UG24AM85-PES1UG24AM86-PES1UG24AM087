"""Build the candidate-level dataset for the training and student-testing partitions.

Part A: candidate-radius analysis on TRAINING trips only - how often the reference
        arc is missing from the candidate set for different radii and K.
Part B: features + labels for every (observation, candidate) pair, saved to
        results/features/{train,student_test}.pkl.

Label quality: an observation's label is trusted only if the GPS point lies within
LABEL_MAX_DIST metres of its aligned reference arc. Untrusted observations are kept
in the files but excluded from training and metrics (column `label_ok`).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.features import DEFAULT_K, DEFAULT_RADIUS, build_trip_frame  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.plotstyle import SERIES, plt, save  # noqa: E402

OUT = ROOT / "results" / "features"
OUT.mkdir(parents=True, exist_ok=True)
LABEL_MAX_DIST = 30.0
RADII = [20, 30, 45, 60, 80, 100]
KS = [3, 5, 8, 10, 15]


def label_rank(trip_dir):
    """Distance to the label arc and its rank among arcs within 100 m, per trusted observation."""
    trip = load_trip(trip_dir)
    net = RoadNetwork(trip.node_xy, trip.arcs)
    labels, d = align_route(trip, net)
    ok = d <= LABEL_MAX_DIST
    ranks = np.full(len(trip.xy), 999)
    for i in np.flatnonzero(ok):
        ids = net.arcs_near(trip.xy[i], 100.0)
        if not len(ids):
            continue
        dd, _ = net.project_points(np.repeat(trip.xy[i][None], len(ids), 0), ids)
        order = ids[np.argsort(dd)]
        hit = np.flatnonzero(order == labels[i])
        if len(hit):
            ranks[i] = hit[0]
    return pd.DataFrame({"trip": trip.trip_id, "label_dist": d[ok], "rank": ranks[ok]})


def build(split, trip_dir):
    t0 = time.time()
    trip = load_trip(trip_dir)
    df, cand, labels, net = build_trip_frame(trip)
    df["label_ok"] = df["align_dist"] <= LABEL_MAX_DIST
    df["split"] = split
    return df, time.time() - t0


def main():
    train_dirs = list_trips(ROOT / "data" / "train")
    test_dirs = list_trips(ROOT / "data" / "student_test")

    # ---- Part A: radius / K analysis (training partition only) ----
    lr = pd.concat(Parallel(n_jobs=8)(delayed(label_rank)(d) for d in train_dirs))
    rows = []
    for r in RADII:
        for k in KS:
            hit = (lr["label_dist"] <= r) & (lr["rank"] < k)
            rows.append({"radius_m": r, "K": k, "candidate_miss_rate": 1 - hit.mean()})
    miss = pd.DataFrame(rows)
    miss.to_csv(OUT / "candidate_miss_analysis_train.csv", index=False)
    print(miss.pivot(index="radius_m", columns="K", values="candidate_miss_rate").round(4))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, k in enumerate(KS):
        s = miss[miss.K == k]
        ax.plot(s.radius_m, s.candidate_miss_rate * 100, marker="o", markersize=5, color=SERIES[i], label=f"K={k}")
    ax.axvline(DEFAULT_RADIUS, color="#52514e", linewidth=1, linestyle="--")
    ax.set(title="Candidate miss rate vs search radius (training trips)", xlabel="radius (m)",
           ylabel="reference arc missing (%)")
    ax.legend(ncol=5)
    save(fig, OUT / "candidate_miss_analysis.png")

    # ---- Part B: features ----
    for split, dirs in (("train", train_dirs), ("student_test", test_dirs)):
        res = Parallel(n_jobs=8)(delayed(build)(split, d) for d in dirs)
        df = pd.concat([r[0] for r in res], ignore_index=True)
        df.to_pickle(OUT / f"{split}.pkl")
        obs = df.groupby(["trip", "obs"]).agg(y=("y", "max"), ok=("label_ok", "first"))
        print(f"{split}: rows={len(df):,} obs={len(obs):,} trusted={obs.ok.mean():.3f} "
              f"miss_rate(trusted)={1 - obs[obs.ok].y.mean():.4f} pos_rate={df.y.mean():.3f} "
              f"build_s={sum(r[1] for r in res):.0f}")


if __name__ == "__main__":
    main()
