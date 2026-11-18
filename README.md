# PathSnap — GPS Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon. PathSnap matches noisy 1 Hz GPS observations to a
road network: a classifier scores candidate road segments for every observation, and an
HMM decoded with Viterbi turns those scores into a connected route. A FastAPI backend
(Laptop 1) serves predictions to a browser UI (Laptop 2).

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | Library code: data loading & projection (`data.py`), road network + bounded Dijkstra (`network.py`), route-to-observation labels (`labels.py`), candidate generation & features (`features.py`), models (`models.py`), metrics (`metrics.py`), HMM/Viterbi (`hmm.py`), end-to-end pipeline (`pipeline.py`), Folium maps (`maps.py`) |
| `scripts/` | Experiments, run in order: `01_eda.py` → `02_build_dataset.py` → `03_train_compare.py` → `04_bayes_opt.py` → `05_hmm_eval.py` |
| `api/` | FastAPI backend (`app.py`) |
| `ui/` | Evaluator UI (`index.html`, Leaflet) |
| `artifacts/` | Final tuned model (`final_model.joblib`) and decoder settings (`hmm_config.json`) |
| `results/` | All experiment outputs: CSVs, plots, Optuna trials, Folium maps |
| `report/` | Technical report |
| `docs/` | Hackathon guideline PDFs |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Place the dataset so that each trip is a folder of `.track`, `.nodes`, `.arcs`, `.route` files:

```
data/train/<trip_id>/...          80 training trips
data/student_test/<trip_id>/...   10 student-testing trips
data/evaluator_test/<trip_id>/... 10 evaluator trips (supplied at evaluation)
```

## Reproduce the experiments

```bash
python scripts/01_eda.py            # dataset exploration -> results/eda/
python scripts/02_build_dataset.py  # candidates + features -> results/features/
python scripts/03_train_compare.py  # 7 models, 5-fold group CV + student-test -> results/models/
python scripts/04_bayes_opt.py      # Optuna TPE on the chosen model -> results/bayes_opt/, artifacts/final_model.joblib
python scripts/05_hmm_eval.py       # decoder tuning + classifier vs HMM -> results/hmm/, results/maps/, artifacts/hmm_config.json
```

Random seed 42 is used throughout. The evaluator-testing partition is never read by any script.

## Run the system (two laptops)

**Laptop 1 — API** (needs this repo, the dataset folder and `artifacts/`):

```bash
set PATHSNAP_DATA_DIR=data\evaluator_test      # optional; defaults to data/evaluator_test, else data/student_test
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Every request is logged in this terminal (trip id, accuracy, timing) for the evaluator.
Find Laptop 1's IP with `ipconfig` and allow port 8000 through the firewall.

**Laptop 2 — UI** (needs only `ui/index.html`):

```bash
cd ui
python -m http.server 5500
```

Open `http://localhost:5500`, enter `http://<LAPTOP1-IP>:8000`, click **Load trips**, then pick a trip.

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | model, decoder settings, data folder |
| GET | `/trips` | – | evaluation trip ids and observation counts |
| POST | `/predict` | `{"trip_id": "00000000"}` | GPS points, predicted route polylines, predicted arc per observation, accuracy metrics, timings |

Reference routes stay on the server: the API uses them only after prediction to compute
accuracy, and never returns them to the UI.

## AI assistance

AI tools (Claude) were used to help write code and documentation. All metrics, tables,
plots and maps in `results/` were produced by running the scripts in this repository.
