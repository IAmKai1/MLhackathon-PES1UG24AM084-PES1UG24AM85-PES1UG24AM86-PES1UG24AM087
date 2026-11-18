"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model.

* Search uses 3-fold GroupKFold on the 80 training trips only (groups = trips).
* Objective: mean CV macro F1 of the candidate classifier (the report's primary
  comparison metric); observation Top-1 accuracy is logged for every trial too.
* Trial 0 is the untuned default configuration, so tuned vs untuned is compared on
  identical folds.
* The best configuration is refit on all training trips and evaluated on the
  student-testing trips. Out-of-fold probabilities for the training trips are saved
  for tuning the HMM decoder without touching test data.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import candidate_metrics, top1_accuracy  # noqa: E402
from pathsnap.models import SEED  # noqa: E402
from pathsnap.plotstyle import SERIES, plt, save  # noqa: E402

FEAT_DIR = ROOT / "results" / "features"
OUT = ROOT / "results" / "bayes_opt"
ART = ROOT / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(exist_ok=True)
N_TRIALS = 25
TRAIN_STRIDE = 3
DEFAULTS = dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20,
                l2_regularization=1e-6, max_features=1.0)


def make_model(params):
    return HistGradientBoostingClassifier(**params, random_state=SEED)


def cv_score(params, train, folds):
    f1s, top1s, lls = [], [], []
    for tr, va in folds:
        d = train.iloc[tr]
        d = d[d.label_ok & (d.obs % TRAIN_STRIDE == 0)]
        m = make_model(params).fit(d[FEATURES].to_numpy(np.float32), d.y.to_numpy())
        v = train.iloc[va]
        v = v[v.label_ok].copy()
        v["p"] = m.predict_proba(v[FEATURES].to_numpy(np.float32))[:, 1]
        cm = candidate_metrics(v.y.to_numpy(), v.p.to_numpy())
        f1s.append(cm["macro_f1"])
        lls.append(cm["log_loss"])
        top1s.append(top1_accuracy(v))
    return float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(top1s)), float(np.mean(lls))


def main():
    train = pd.read_pickle(FEAT_DIR / "train.pkl")
    test = pd.read_pickle(FEAT_DIR / "student_test.pkl")
    folds = list(GroupKFold(n_splits=3).split(train, groups=train.trip.to_numpy()))

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 600, step=50),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
        )
        t0 = time.time()
        f1, f1_std, top1, ll = cv_score(params, train, folds)
        trial.set_user_attr("cv_macro_f1_std", f1_std)
        trial.set_user_attr("cv_top1_acc", top1)
        trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number:2d}: f1={f1:.4f} top1={top1:.4f} ll={ll:.4f} {params}", flush=True)
        return f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="hgb_pathsnap")
    study.enqueue_trial(DEFAULTS)                      # trial 0 = untuned baseline
    study.optimize(objective, n_trials=N_TRIALS)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = study.best_params
    json.dump({"best_params": best, "best_cv_macro_f1": study.best_value, "n_trials": N_TRIALS,
               "cv": "3-fold GroupKFold over 80 training trips", "objective": "mean CV macro F1",
               "sampler": "Optuna TPESampler(seed=42)"}, open(OUT / "best_params.json", "w"), indent=2)

    # Tuned vs untuned on the student-testing trips (same training data, same metrics)
    rows = []
    for label, params in (("untuned (defaults)", DEFAULTS), ("tuned (Bayesian)", best)):
        d = train[train.label_ok & (train.obs % TRAIN_STRIDE == 0)]
        t0 = time.time()
        m = make_model(params).fit(d[FEATURES].to_numpy(np.float32), d.y.to_numpy())
        fit_s = time.time() - t0
        te = test.copy()
        t0 = time.time()
        te["p"] = m.predict_proba(te[FEATURES].to_numpy(np.float32))[:, 1]
        inf_s = time.time() - t0
        tm = te[te.label_ok]
        cm = candidate_metrics(tm.y.to_numpy(), tm.p.to_numpy())
        trial_row = trials.iloc[0] if label.startswith("untuned") else trials.loc[trials.value.idxmax()]
        rows.append({"model": label, "cv_macro_f1": trial_row["value"],
                     "cv_top1_acc": trial_row["user_attrs_cv_top1_acc"],
                     "test_top1_acc": top1_accuracy(te), "test_macro_f1": cm["macro_f1"],
                     "test_log_loss": cm["log_loss"], "train_time_s": fit_s, "inference_time_s": inf_s})
        if label.startswith("tuned"):
            joblib.dump(m, ART / "final_model.joblib", compress=3)
            np.save(OUT / "student_test_proba_tuned.npy", te.p.to_numpy())
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print(comp.round(4).to_string())

    # Out-of-fold probabilities on training trips (for HMM decoder tuning)
    oof = np.full(len(train), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(train, groups=train.trip.to_numpy()):
        d = train.iloc[tr]
        d = d[d.label_ok & (d.obs % TRAIN_STRIDE == 0)]
        m = make_model(best).fit(d[FEATURES].to_numpy(np.float32), d.y.to_numpy())
        oof[va] = m.predict_proba(train.iloc[va][FEATURES].to_numpy(np.float32))[:, 1]
    np.save(OUT / "train_oof_proba_tuned.npy", oof)

    # Plots
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(trials.number, trials.value, marker="o", markersize=5, linewidth=0, color=SERIES[0], label="trial")
    ax.plot(trials.number, trials.value.cummax(), color=SERIES[1], label="best so far")
    ax.axhline(trials.value.iloc[0], color="#52514e", linewidth=1, linestyle="--")
    ax.text(len(trials) - 1, trials.value.iloc[0], "untuned", ha="right", va="bottom", fontsize=7, color="#52514e")
    ax.set(title="Bayesian optimisation history (3-fold group CV macro F1)", xlabel="trial", ylabel="macro F1")
    ax.legend(ncol=2, loc="lower right")
    save(fig, OUT / "optimisation_history.png")

    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1], color=SERIES[0], height=0.6)
        ax.set(title="Hyperparameter importance (fANOVA)", xlabel="importance")
        save(fig, OUT / "param_importance.png")
    except Exception as e:  # importance needs enough completed trials
        print("param importance skipped:", e)


if __name__ == "__main__":
    main()
