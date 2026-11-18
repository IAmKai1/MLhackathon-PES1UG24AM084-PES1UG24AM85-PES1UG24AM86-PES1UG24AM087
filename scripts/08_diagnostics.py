"""Extra diagnostics for the report (training and student-testing data only).

1. Overfitting check: training-set vs CV vs student-test macro F1 / Top-1 for every model.
2. Training and validation loss curves for the iterative models:
   - HistGradientBoosting (tuned): log loss after each boosting iteration (staged predictions)
   - MLP: log loss after each epoch (partial_fit)
   Both use CV fold 0 of the training trips: fit on its training trips, validate on its
   held-out trips. No test data is involved.
3. HMM error analysis on student-test: observations the HMM fixed vs observations it broke.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.hmm import HMMConfig  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.metrics import candidate_metrics, top1_accuracy  # noqa: E402
from pathsnap.models import MODELS, SEED  # noqa: E402
from pathsnap.pipeline import LABEL_MAX_DIST, run_trip  # noqa: E402
from pathsnap.plotstyle import SERIES, plt, save  # noqa: E402

FEAT = ROOT / "results" / "features"
OUT = ROOT / "results" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
STRIDE = 3


def overfitting_table(train):
    """Training-set scores of the saved full-train models, next to their CV and test scores."""
    comp = pd.read_csv(ROOT / "results" / "models" / "model_comparison.csv")
    tr = train[train.label_ok & (train.obs % STRIDE == 0)]
    # Score on a fixed subsample of the training rows (same rows for every model)
    obs_keep = tr[["trip", "obs"]].drop_duplicates().sample(20000, random_state=SEED)
    tr = tr.merge(obs_keep, on=["trip", "obs"])
    rows = []
    for name in MODELS:
        model = joblib.load(ROOT / "results" / "models" / f"model_{name}.joblib")
        d = tr.copy()
        d["p"] = model.predict_proba(d[FEATURES].to_numpy(np.float32))[:, 1]
        m = candidate_metrics(d.y.to_numpy(), d.p.to_numpy())
        c = comp[comp.model == name].iloc[0]
        rows.append({"model": name, "train_macro_f1": m["macro_f1"], "cv_macro_f1": c.cv_macro_f1,
                     "test_macro_f1": c.test_macro_f1, "train_top1": top1_accuracy(d),
                     "cv_top1": c.cv_top1_acc, "test_top1": c.test_top1_acc,
                     "train_log_loss": m["log_loss"], "test_log_loss": c.test_log_loss,
                     "train_minus_cv_f1": m["macro_f1"] - c.cv_macro_f1})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "train_vs_cv_vs_test.csv", index=False)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (col, lab) in enumerate([("train_macro_f1", "Training"), ("cv_macro_f1", "5-fold group CV"),
                                    ("test_macro_f1", "Student-test")]):
        ax.bar(x + (i - 1) * 0.27, df[col], 0.25, color=SERIES[i], label=lab)
    ax.set_xticks(x, df.model, rotation=20, ha="right")
    ax.set(title="Macro F1 on training data vs CV vs student-test (overfitting check)", ylabel="macro F1",
           ylim=(0.8, 1.0))
    ax.legend(ncol=3, loc="upper left")
    save(fig, OUT / "train_vs_cv_vs_test.png")


def loss_curves(train):
    tr_idx, va_idx = next(GroupKFold(n_splits=5).split(train, groups=train.trip.to_numpy()))
    tr = train.iloc[tr_idx]
    tr = tr[tr.label_ok & (tr.obs % STRIDE == 0)]
    va = train.iloc[va_idx]
    va = va[va.label_ok]
    Xtr, ytr = tr[FEATURES].to_numpy(np.float32), tr.y.to_numpy()
    Xva, yva = va[FEATURES].to_numpy(np.float32), va.y.to_numpy()

    best = json.load(open(ROOT / "results" / "bayes_opt" / "best_params.json"))["best_params"]
    hgb = HistGradientBoostingClassifier(**best, random_state=SEED).fit(Xtr, ytr)
    hgb_tr = [log_loss(ytr, p[:, 1]) for p in hgb.staged_predict_proba(Xtr)]
    hgb_va = [log_loss(yva, p[:, 1]) for p in hgb.staged_predict_proba(Xva)]
    pd.DataFrame({"iteration": np.arange(1, len(hgb_tr) + 1), "train_log_loss": hgb_tr,
                  "val_log_loss": hgb_va}).to_csv(OUT / "loss_curve_HistGradientBoosting.csv", index=False)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
                        random_state=SEED)
    mlp_tr, mlp_va = [], []
    for _ in range(30):
        mlp.partial_fit(Xtr_s, ytr, classes=[0, 1])
        mlp_tr.append(log_loss(ytr, mlp.predict_proba(Xtr_s)[:, 1]))
        mlp_va.append(log_loss(yva, mlp.predict_proba(Xva_s)[:, 1]))
    pd.DataFrame({"epoch": np.arange(1, 31), "train_log_loss": mlp_tr, "val_log_loss": mlp_va}).to_csv(
        OUT / "loss_curve_MLP.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (tl, vl, xlab, title) in zip(axes, [(hgb_tr, hgb_va, "boosting iteration", "HistGradientBoosting (tuned)"),
                                               (mlp_tr, mlp_va, "epoch", "MLP (64, 32)")]):
        ax.plot(np.arange(1, len(tl) + 1), tl, color=SERIES[0], label="training loss")
        ax.plot(np.arange(1, len(vl) + 1), vl, color=SERIES[1], label="validation loss")
        best_i = int(np.argmin(vl)) + 1
        ax.axvline(best_i, color="#52514e", linewidth=1, linestyle="--")
        ax.set(title=f"{title}: log loss (val min at {best_i})", xlabel=xlab, ylabel="log loss")
        ax.legend()
    save(fig, OUT / "training_validation_loss.png")
    print("HGB final train/val:", hgb_tr[-1], hgb_va[-1], "| MLP final train/val:", mlp_tr[-1], mlp_va[-1])


def hmm_error_analysis():
    model = joblib.load(ROOT / "artifacts" / "final_model.joblib")
    raw = json.load(open(ROOT / "artifacts" / "hmm_config.json"))
    cfg = HMMConfig(**{k: raw[k] for k in HMMConfig().to_dict()})
    rows = []
    for trip_dir in sorted((ROOT / "data" / "student_test").iterdir()):
        r = run_trip(trip_dir, model, cfg)
        trip, net = r["trip"], r["net"]
        labels, d = align_route(trip, net)
        ok = d <= LABEL_MAX_DIST
        c, h, lab = r["classifier_arcs"][ok], r["hmm_arcs"][ok], labels[ok]
        rows.append({"trip": trip_dir.name, "scored_obs": int(ok.sum()),
                     "both_correct": int(((c == lab) & (h == lab)).sum()),
                     "hmm_fixed": int(((c != lab) & (h == lab)).sum()),
                     "hmm_broke": int(((c == lab) & (h != lab)).sum()),
                     "both_wrong": int(((c != lab) & (h != lab)).sum())})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.loc["total"] = ["TOTAL"] + df.iloc[:, 1:].sum().tolist()
    df.to_csv(OUT / "hmm_fixed_vs_broke_student_test.csv", index=False)


def main():
    train = pd.read_pickle(FEAT / "train.pkl")
    overfitting_table(train)
    loss_curves(train)
    hmm_error_analysis()


if __name__ == "__main__":
    main()
