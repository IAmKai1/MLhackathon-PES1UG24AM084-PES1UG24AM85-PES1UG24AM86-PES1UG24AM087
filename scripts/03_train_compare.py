"""Train and compare the candidate classifiers.

* 5-fold GroupKFold on the 80 training trips (groups = trips, so no trip is split
  across folds). Training folds use every 3rd observation per trip (1 Hz points are
  strongly autocorrelated); validation folds use all trusted observations.
* Each model is then refit on all training trips and evaluated on the 10
  student-testing trips (development-stage comparison only).

Outputs: results/models/*.csv, plots, fitted models and student-test probabilities.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_trip  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import candidate_metrics, route_continuity, top1_accuracy, top1_predictions  # noqa: E402
from pathsnap.models import MODELS, SEED  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.plotstyle import SERIES, plt, save  # noqa: E402

FEAT_DIR = ROOT / "results" / "features"
OUT = ROOT / "results" / "models"
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_STRIDE = 3


def train_rows(df):
    return df[df.label_ok & (df.obs % TRAIN_STRIDE == 0)]


def fit(name, df):
    family, factory, cap, _ = MODELS[name]
    d = train_rows(df)
    if cap and len(d) > cap:
        d = d.sample(cap, random_state=SEED)
    model = factory()
    t0 = time.time()
    model.fit(d[FEATURES].to_numpy(np.float32), d.y.to_numpy())
    return model, time.time() - t0, len(d)


def predict(model, df):
    t0 = time.time()
    p = model.predict_proba(df[FEATURES].to_numpy(np.float32))[:, 1]
    return p, time.time() - t0


def main():
    train = pd.read_pickle(FEAT_DIR / "train.pkl")
    test = pd.read_pickle(FEAT_DIR / "student_test.pkl")
    groups = train.trip.to_numpy()
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train, groups=groups))
    json.dump({"folds": [sorted(train.trip.iloc[v].unique().tolist()) for _, v in folds]},
              open(OUT / "cv_folds.json", "w"), indent=1)

    test_nets = {tid: RoadNetwork(load_trip(ROOT / "data" / "student_test" / tid).node_xy,
                                  load_trip(ROOT / "data" / "student_test" / tid).arcs)
                 for tid in test.trip.unique()}

    fold_rows, summary = [], []
    for name in MODELS:
        family, _, _, config = MODELS[name]
        print(f"=== {name}")
        for k, (tr, va) in enumerate(folds):
            model, fit_s, n_fit = fit(name, train.iloc[tr])
            va_df = train.iloc[va]
            va_df = va_df[va_df.label_ok].copy()
            va_df["p"], pred_s = predict(model, va_df)
            m = candidate_metrics(va_df.y.to_numpy(), va_df.p.to_numpy())
            m.update(model=name, fold=k, top1_acc=top1_accuracy(va_df), fit_s=fit_s, n_fit_rows=n_fit)
            fold_rows.append(m)
            print(f"  fold {k}: macroF1={m['macro_f1']:.4f} top1={m['top1_acc']:.4f} logloss={m['log_loss']:.4f} fit={fit_s:.0f}s")

        # Refit on all training trips, evaluate on student-testing trips
        model, fit_s, n_fit = fit(name, train)
        te = test.copy()
        te["p"], pred_s = predict(model, te)
        np.save(OUT / f"student_test_proba_{name}.npy", te.p.to_numpy())
        joblib.dump(model, OUT / f"model_{name}.joblib", compress=3)
        tm = te[te.label_ok]
        mt = candidate_metrics(tm.y.to_numpy(), tm.p.to_numpy())
        top = top1_predictions(te)
        cont = np.mean([route_continuity(g.sort_values("obs").arc.to_numpy(), test_nets[tid])
                        for tid, g in top.groupby("trip")])
        if hasattr(model, "loss_curve_") or (hasattr(model, "steps") and hasattr(model[-1], "loss_curve_")):
            mlp = model[-1] if hasattr(model, "steps") else model
            pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1), "train_loss": mlp.loss_curve_,
                          "val_accuracy": mlp.validation_scores_}).to_csv(OUT / f"training_history_{name}.csv", index=False)
        cvf = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        summary.append({
            "model": name, "family": family, "config": config,
            "cv_macro_f1": cvf.macro_f1.mean(), "cv_macro_f1_std": cvf.macro_f1.std(),
            "cv_top1_acc": cvf.top1_acc.mean(), "cv_log_loss": cvf.log_loss.mean(),
            "test_macro_f1": mt["macro_f1"], "test_macro_precision": mt["macro_precision"],
            "test_macro_recall": mt["macro_recall"], "test_log_loss": mt["log_loss"],
            "test_top1_acc": top1_accuracy(te), "test_route_continuity": cont,
            "train_time_s": fit_s, "n_train_rows": n_fit,
            "inference_time_s": pred_s, "inference_ms_per_obs": 1000 * pred_s / te.groupby(["trip", "obs"]).ngroups,
        })
        print(f"  student-test: macroF1={mt['macro_f1']:.4f} top1={summary[-1]['test_top1_acc']:.4f} "
              f"continuity={cont:.3f} logloss={mt['log_loss']:.4f}")

        if name == "HistGradientBoosting":
            cm = confusion_matrix(tm.y, (tm.p >= 0.5).astype(int))
            fig, ax = plt.subplots(figsize=(4, 3.6))
            ConfusionMatrixDisplay(cm, display_labels=["not ref arc", "ref arc"]).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.grid(False)
            ax.set_title("HistGradientBoosting - student-test candidates")
            save(fig, OUT / "confusion_matrix_HistGradientBoosting.png")

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(OUT / "cv_fold_results.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "model_comparison.csv", index=False)
    print(summ[["model", "cv_macro_f1", "cv_top1_acc", "test_macro_f1", "test_top1_acc",
                "test_log_loss", "test_route_continuity", "inference_ms_per_obs"]].round(4).to_string())

    # ---- comparison plots ----
    names = summ.model.tolist()
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x, summ.cv_macro_f1, yerr=summ.cv_macro_f1_std, color=SERIES[0], width=0.6, capsize=3,
           error_kw={"elinewidth": 1, "ecolor": "#52514e"})
    for i, v in enumerate(summ.cv_macro_f1):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, color="#52514e")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set(title="5-fold group CV macro F1 (training trips; error bar = fold std)", ylabel="macro F1",
           ylim=(0, 1.05))
    save(fig, OUT / "cv_macro_f1.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, summ.test_top1_acc, w, color=SERIES[0], label="Top-1 accuracy")
    ax.bar(x + w / 2 + 0.01, summ.test_macro_f1, w, color=SERIES[1], label="Macro F1")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set(title="Student-testing trips: Top-1 accuracy and macro F1", ylabel="score", ylim=(0, 1.05))
    ax.legend(ncol=2, loc="lower right")
    save(fig, OUT / "student_test_top1_f1.png")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(x, summ.test_log_loss, color=SERIES[0], width=0.6)
    for i, v in enumerate(summ.test_log_loss):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#52514e")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set(title="Student-testing log loss (lower is better)", ylabel="log loss")
    save(fig, OUT / "student_test_log_loss.png")

    hist = OUT / "training_history_MLP.csv"
    if hist.exists():
        h = pd.read_csv(hist)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        axes[0].plot(h.epoch, h.train_loss, color=SERIES[0])
        axes[0].set(title="MLP training loss (log loss)", xlabel="epoch", ylabel="loss")
        axes[1].plot(h.epoch, h.val_accuracy, color=SERIES[1])
        axes[1].set(title="MLP validation accuracy (10% hold-out)", xlabel="epoch", ylabel="accuracy")
        save(fig, OUT / "mlp_training_history.png")


if __name__ == "__main__":
    main()
