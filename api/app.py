"""PathSnap inference API (runs on Laptop 1).

Endpoints
  GET  /health            model + decoder status
  GET  /trips             trip ids available in the evaluation data folder
  POST /predict           {"trip_id": "..."} -> predicted route + server-side evaluation

The evaluation folder is set with PATHSNAP_DATA_DIR (default: data/evaluator_test if it
exists, else data/student_test). Reference routes in that folder are read only on the
server, after prediction, to compute accuracy; they are never returned to the UI.

Run:  uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import unproject  # noqa: E402
from pathsnap.hmm import HMMConfig  # noqa: E402
from pathsnap.maps import dedupe_consecutive  # noqa: E402
from pathsnap.pipeline import run_trip  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

default_dir = ROOT / "data" / "evaluator_test"
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR",
                               default_dir if default_dir.exists() else ROOT / "data" / "student_test"))
MODEL = joblib.load(ROOT / "artifacts" / "final_model.joblib")
_hmm = json.load(open(ROOT / "artifacts" / "hmm_config.json"))
HMM_CFG = HMMConfig(**{k: _hmm[k] for k in HMMConfig().to_dict()})
log.info("Model loaded: %s | decoder: %s | data: %s", type(MODEL).__name__, HMM_CFG.to_dict(), DATA_DIR)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _trip_ids():
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


@app.get("/")
def root():
    return {"service": "PathSnap API", "endpoints": ["/health", "/trips", "/predict", "/docs"]}


@app.get("/health")
def health():
    return {"status": "ok", "model": type(MODEL).__name__, "decoder": HMM_CFG.to_dict(),
            "data_dir": str(DATA_DIR), "n_trips": len(_trip_ids())}


@app.get("/trips")
def trips():
    out = []
    for tid in _trip_ids():
        track = DATA_DIR / tid / f"{tid}.track"
        with open(track) as f:
            n = sum(1 for _ in f)
        out.append({"trip_id": tid, "n_observations": n})
    log.info("GET /trips -> %d trips", len(out))
    return {"trips": out}


def _route_lines(trip, arcs):
    """Predicted route as lon/lat polylines, split wherever consecutive arcs do not connect."""
    lines, cur, prev_head = [], [], None
    for a in dedupe_consecutive(arcs[arcs >= 0]):
        u, v = trip.arcs[a]
        if prev_head is not None and u != prev_head and cur:
            lines.append(cur)
            cur = []
        if not cur:
            cur.append([float(trip.nodes[u, 1]), float(trip.nodes[u, 0])])
        cur.append([float(trip.nodes[v, 1]), float(trip.nodes[v, 0])])
        prev_head = v
    if cur:
        lines.append(cur)
    return lines


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id
    if tid not in _trip_ids():
        raise HTTPException(404, f"unknown trip {tid}")
    log.info("POST /predict trip=%s : received", tid)
    t0 = time.time()
    r = run_trip(DATA_DIR / tid, MODEL, HMM_CFG, score=True)
    trip = r["trip"]
    elapsed = time.time() - t0
    sc, sh = r.get("score_classifier", {}), r.get("score_hmm", {})
    stride = max(1, len(trip.track) // 2500)
    resp = {
        "trip_id": tid,
        "n_observations": r["n_observations"],
        "gps": trip.track[::stride, [1, 0]].round(6).tolist(),
        "predicted_route": _route_lines(trip, r["hmm_arcs"]),
        "predicted_arc_per_observation": r["hmm_arcs"].tolist(),
        "metrics": {
            "observation_accuracy_hmm": sh.get("observation_accuracy"),
            "observation_accuracy_classifier_only": sc.get("observation_accuracy"),
            "route_set_f1_hmm": sh.get("route_set_f1"),
            "route_continuity_hmm": r["continuity_hmm"],
            "route_continuity_classifier_only": r["continuity_classifier"],
            "trusted_observations": sh.get("trusted_observations"),
            "chain_breaks": r["chain_breaks"],
        },
        "timing_s": {"features": round(r["time_features_s"], 3), "classify_and_decode": round(r["time_decode_s"], 3),
                     "total": round(elapsed, 3)},
    }
    log.info("POST /predict trip=%s : %d obs, HMM accuracy=%.4f (classifier-only %.4f), continuity=%.3f, %.2fs",
             tid, r["n_observations"], sh.get("observation_accuracy") or float("nan"),
             sc.get("observation_accuracy") or float("nan"), r["continuity_hmm"], elapsed)
    return resp
