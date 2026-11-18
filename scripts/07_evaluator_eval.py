"""Final evaluation on the 10 evaluator-testing trips.

Inference only: the saved final model (artifacts/final_model.joblib) and decoder settings
(artifacts/hmm_config.json) were fixed before this data arrived and are not changed here.
Nothing in this script trains, tunes or selects anything. It runs the same pipeline the
API uses and records the results.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.hmm import HMMConfig  # noqa: E402
from pathsnap.maps import trip_map  # noqa: E402
from pathsnap.pipeline import run_trip  # noqa: E402

DATA = ROOT / "data" / "evaluator_test"
OUT = ROOT / "results" / "evaluator"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    model = joblib.load(ROOT / "artifacts" / "final_model.joblib")
    raw = json.load(open(ROOT / "artifacts" / "hmm_config.json"))
    cfg = HMMConfig(**{k: raw[k] for k in HMMConfig().to_dict()})
    rows = []
    for trip_dir in sorted(DATA.iterdir()):
        r = run_trip(trip_dir, model, cfg)
        sc, sh = r["score_classifier"], r["score_hmm"]
        rows.append({
            "trip": trip_dir.name, "n_obs": r["n_observations"], "trusted_obs": sh["trusted_observations"],
            "chain_breaks": r["chain_breaks"],
            "clf_obs_acc": sc["observation_accuracy"], "hmm_obs_acc": sh["observation_accuracy"],
            "clf_continuity": r["continuity_classifier"], "hmm_continuity": r["continuity_hmm"],
            "clf_route_f1": sc["route_set_f1"], "hmm_route_f1": sh["route_set_f1"],
            "seconds": r["time_features_s"] + r["time_decode_s"],
        })
        print(rows[-1], flush=True)
        trip_map(r["trip"], r["net"], OUT / f"evaluator_{trip_dir.name}.html", predicted_arcs=r["hmm_arcs"],
                 title=f"Evaluator trip {trip_dir.name}: HMM accuracy {sh['observation_accuracy']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "evaluator_classifier_vs_hmm.csv", index=False)
    w = res.trusted_obs / res.trusted_obs.sum()
    summary = {
        "classifier_obs_acc_weighted": float((res.clf_obs_acc * w).sum()),
        "hmm_obs_acc_weighted": float((res.hmm_obs_acc * w).sum()),
        "classifier_continuity_mean": float(res.clf_continuity.mean()),
        "hmm_continuity_mean": float(res.hmm_continuity.mean()),
        "classifier_route_f1_mean": float(res.clf_route_f1.mean()),
        "hmm_route_f1_mean": float(res.hmm_route_f1.mean()),
        "trusted_fraction": float(res.trusted_obs.sum() / res.n_obs.sum()),
        "model": "artifacts/final_model.joblib (unchanged)", "decoder": cfg.to_dict(),
    }
    json.dump(summary, open(OUT / "evaluator_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
