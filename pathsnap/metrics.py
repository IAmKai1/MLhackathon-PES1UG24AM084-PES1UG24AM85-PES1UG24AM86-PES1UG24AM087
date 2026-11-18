"""Candidate-level and observation-level evaluation metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def top1_predictions(df, score_col="p"):
    """Highest-scoring candidate per observation -> DataFrame(trip, obs, arc, label_arc, y)."""
    idx = df.groupby(["trip", "obs"], sort=True)[score_col].idxmax()
    return df.loc[idx, ["trip", "obs", "arc", "label_arc", "y", "label_ok"]].reset_index(drop=True)


def top1_accuracy(df, score_col="p"):
    top = top1_predictions(df, score_col)
    top = top[top.label_ok]
    return float(top.y.mean())


def candidate_miss_rate(df):
    obs = df[df.label_ok].groupby(["trip", "obs"]).y.max()
    return float(1 - obs.mean())


def route_continuity(pred_arcs, net):
    """Share of consecutive observation pairs whose predicted arcs are the same or directly connected."""
    a, b = np.asarray(pred_arcs[:-1]), np.asarray(pred_arcs[1:])
    if len(a) == 0:
        return 1.0
    ok = (a == b) | (net.arcs[a, 1] == net.arcs[b, 0])
    return float(ok.mean())


def route_set_f1(pred_arcs, ref_arcs):
    """Sequence-level score: F1 between the sets of distinct arcs predicted and in the reference route."""
    p, r = set(np.asarray(pred_arcs).tolist()), set(np.asarray(ref_arcs).tolist())
    if not p or not r:
        return 0.0
    inter = len(p & r)
    prec, rec = inter / len(p), inter / len(r)
    return 0.0 if inter == 0 else 2 * prec * rec / (prec + rec)


def summarise_trip_predictions(trip_df_pred, net, ref_route):
    """Per-trip observation accuracy (trusted labels), continuity and route-set F1."""
    d = trip_df_pred.sort_values("obs")
    trusted = d[d.label_ok]
    return {
        "top1_acc": float(trusted.y.mean()) if len(trusted) else np.nan,
        "continuity": route_continuity(d.arc.to_numpy(), net),
        "route_f1": route_set_f1(d.arc.to_numpy(), ref_route),
    }
