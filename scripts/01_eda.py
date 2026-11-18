"""Dataset exploration: inventory, sampling, gaps, extent, network connectivity and
reference-route agreement for every trip in the training and student-testing partitions.

Outputs go to results/eda/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.maps import trip_map  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.plotstyle import SERIES, plt, save  # noqa: E402

OUT = ROOT / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def route_agreement(trip, net, chunk=4000):
    """Distance from each GPS point to the nearest arc of the reference route."""
    a, b = net.a[trip.route], net.b[trip.route]
    ab = b - a
    l2 = np.maximum((ab ** 2).sum(1), 1e-9)
    out = []
    for s in range(0, len(trip.xy), chunk):
        p = trip.xy[s:s + chunk, None, :]
        t = np.clip(((p - a) * ab).sum(-1) / l2, 0, 1)
        q = a + t[..., None] * ab
        out.append(np.hypot(*(q - p).transpose(2, 0, 1)).min(axis=1))
    return np.concatenate(out)


def trip_stats(split, trip_dir):
    tid = trip_dir.name
    raw = np.loadtxt(trip_dir / f"{tid}.track", ndmin=2)
    trip = load_trip(trip_dir)
    net = RoadNetwork(trip.node_xy, trip.arcs)
    dt = np.diff(trip.t)
    step = np.hypot(*np.diff(trip.xy, axis=0).T)
    A, M = len(trip.arcs), len(trip.nodes)
    g = coo_matrix((np.ones(A), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(M, M))
    n_comp, comp = connected_components(g, directed=True, connection="weak")
    sizes = np.bincount(comp)
    deg = net.out_deg + net.in_deg
    row = dict(
        split=split, trip=tid, n_obs=len(trip.track), raw_rows=len(raw),
        invalid_rows=int(len(raw) - np.isfinite(raw).all(1).sum()),
        unsorted_steps=int((np.diff(raw[:, 2]) <= 0).sum()),
        duration_s=float(trip.t[-1] - trip.t[0]), frac_dt_1s=float(np.mean(np.isclose(dt, 1.0))),
        gaps_gt5s=int((dt > 5).sum()), max_gap_s=float(dt.max()) if len(dt) else 0.0,
        path_km=float(step.sum() / 1000), median_speed_ms=float(np.median(step / np.maximum(dt, 1e-6))),
        extent_x_km=float(np.ptp(trip.xy[:, 0]) / 1000), extent_y_km=float(np.ptp(trip.xy[:, 1]) / 1000),
        lon_min=float(trip.track[:, 0].min()), lon_max=float(trip.track[:, 0].max()),
        lat_min=float(trip.track[:, 1].min()), lat_max=float(trip.track[:, 1].max()),
        n_nodes=M, n_arcs=A, isolated_nodes=int((deg == 0).sum()), weak_components=int(n_comp),
        largest_component_frac=float(sizes.max() / M),
        route_len=len(trip.route) if trip.route is not None else 0,
    )
    if trip.route is not None and len(trip.route):
        r = trip.route
        row["route_connected_frac"] = float(np.mean(trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0])) if len(r) > 1 else 1.0
        row["route_km"] = float(net.length[r].sum() / 1000)
        d = route_agreement(trip, net)
        row["gps_to_route_median_m"] = float(np.median(d))
        row["gps_within_30m_of_route"] = float(np.mean(d <= 30))
    return row, trip, net


def main():
    rows = []
    examples = {}
    for split in ("train", "student_test"):
        for trip_dir in list_trips(ROOT / "data" / split):
            row, trip, net = trip_stats(split, trip_dir)
            rows.append(row)
            if trip_dir.name in ("00000003", "00000054", "00000007"):
                examples[trip_dir.name] = (trip, net)
            print(f"{split:12s} {row['trip']} obs={row['n_obs']:6d} arcs={row['n_arcs']:7d} "
                  f"within30m={row.get('gps_within_30m_of_route', float('nan')):.2f}")
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "trip_inventory.csv", index=False)

    summary = inv.groupby("split").agg(
        trips=("trip", "count"), obs_total=("n_obs", "sum"), obs_median=("n_obs", "median"),
        obs_min=("n_obs", "min"), obs_max=("n_obs", "max"), invalid_rows=("invalid_rows", "sum"),
        unsorted_steps=("unsorted_steps", "sum"), frac_dt_1s_mean=("frac_dt_1s", "mean"),
        trips_with_gap_gt5s=("gaps_gt5s", lambda s: int((s > 0).sum())), max_gap_s=("max_gap_s", "max"),
        arcs_median=("n_arcs", "median"), isolated_nodes=("isolated_nodes", "sum"),
        weak_components_median=("weak_components", "median"),
        route_agreement_median=("gps_within_30m_of_route", "median"),
        trips_poor_route_agreement=("gps_within_30m_of_route", lambda s: int((s < 0.8).sum())),
    ).T
    summary.to_csv(OUT / "summary_by_split.csv")
    print(summary)

    # --- plots ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, split in enumerate(("train", "student_test")):
        axes[0].hist(inv.loc[inv.split == split, "n_obs"], bins=25, alpha=0.85, color=SERIES[i], label=split,
                     edgecolor="#fcfcfb", linewidth=1)
    axes[0].set(title="GPS observations per trip", xlabel="observations", ylabel="trips")
    axes[0].legend()
    axes[1].hist(inv["max_gap_s"], bins=30, color=SERIES[0], edgecolor="#fcfcfb", linewidth=1)
    axes[1].set(title="Largest time gap per trip", xlabel="seconds", ylabel="trips")
    axes[2].hist(inv["gps_within_30m_of_route"], bins=20, color=SERIES[0], edgecolor="#fcfcfb", linewidth=1)
    axes[2].axvline(0.8, color="#52514e", linewidth=1, linestyle="--")
    axes[2].set(title="Share of GPS points within 30 m of route", xlabel="fraction", ylabel="trips")
    save(fig, OUT / "trip_distributions.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, split in enumerate(("train", "student_test")):
        for tid in inv.loc[inv.split == split, "trip"]:
            tr = np.loadtxt(ROOT / "data" / split / tid / f"{tid}.track", ndmin=2)
            ax.plot(tr[::5, 0], tr[::5, 1], color=SERIES[i], linewidth=0.6, alpha=0.6)
        ax.plot([], [], color=SERIES[i], label=split)
    ax.set(title="All GPS trajectories (lon/lat)", xlabel="longitude", ylabel="latitude")
    ax.legend()
    save(fig, OUT / "all_trajectories.png")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    dts = []
    for split in ("train", "student_test"):
        for tid in inv.loc[inv.split == split, "trip"]:
            tr = np.loadtxt(ROOT / "data" / split / tid / f"{tid}.track", ndmin=2)
            dts.append(np.diff(tr[:, 2]))
    dts = np.concatenate(dts)
    vals, counts = np.unique(np.clip(np.round(dts), 0, 10), return_counts=True)
    ax.bar(vals, counts, color=SERIES[0], width=0.7)
    ax.set_yscale("log")
    ax.set(title="Sampling interval between consecutive GPS points (10 = 10 s or more)",
           xlabel="seconds", ylabel="count (log)")
    save(fig, OUT / "sampling_intervals.png")

    for tid, (trip, net) in examples.items():
        trip_map(trip, net, OUT / f"map_trip_{tid}.html", title=f"Trip {tid}: GPS vs reference route")
    print("EDA written to", OUT)


if __name__ == "__main__":
    main()
