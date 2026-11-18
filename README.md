# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap snaps noisy GPS trajectories onto a road network and reconstructs the route that was driven. A classifier scores each GPS point's candidate road segments (arcs), and an HMM with Viterbi decoding turns those scores into one connected route. The system runs on two laptops: a FastAPI inference server and a Streamlit evaluator UI that calls it over the LAN.

## Team

| Name | SRN | GitHub |
| --- | --- | --- |
| Maaya Mohan | PES2UG24CS254 | [@maayamohan](https://github.com/maayamohan) |
| Manasi Dinesh Unnithan | PES2UG24CS260 | [@idwmu](https://github.com/idwmu) |
| Marella Likhita Sri Durga | PES2UG24CS269 | [@likhita-marella](https://github.com/likhita-marella) |
| Manoj Kumar R | PES2UG24CS267 | [@ColdHeart-35](https://github.com/ColdHeart-35) |

## How it works

```
GPS track (.track) + road network (.nodes, .arcs)
        |
        v
data_processing/   clean -> project to UTM metres -> k=6 nearest candidate arcs per point -> features
        |
        v
models/model.pkl   HistGradientBoosting: P(candidate is the true arc) per (point, candidate)
        |
        v
backend/hmm/       emissions (classifier probs) + transitions (road connectivity, GPS gaps) -> Viterbi
        |
        v
backend/ (FastAPI) GET /trips, POST /predict -> predicted arc sequence + route F1 / precision / recall
        |
        v
ui/app.py          Streamlit + Folium: raw GPS (red), matched route (blue), reference route (green)
```

Ground-truth routes stay on the server. The API computes route accuracy internally and returns only the metrics.

## Repository layout

| Path | Contents |
| --- | --- |
| `datasets/` | `train/` (80 trips), `student_test/` (10), `evaluator_test/` (10). Each trip folder has `.track`, `.nodes`, `.arcs`, `.route`. |
| `data_processing/` | Loaders, cleaning, UTM projection, candidate generation, labelling, features; `eda.ipynb`, `folium_maps.ipynb` |
| `models/` | Model comparison, tuning, final training, report plots, HMM experiment; `model.pkl`, `model_tuned.pkl` |
| `backend/` | FastAPI app (`main.py`), routes, HMM (`emissions`, `transitions`, `viterbi`), inference pipeline, Pydantic schemas |
| `ui/app.py` | Streamlit evaluator UI |
| `tests/unit/` | Backend unit tests (emissions, transitions, Viterbi, model loader) |
| `results/` | Feature tables (`features_*.parquet`), EDA outputs (`eda/`, `plots/`), report figures (`figures/`), Folium maps (`folium/`) |
| `SYSTEM_C_README.md` | Detailed backend notes |

## Setup

Python 3.13, one virtual environment per laptop.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pyproj shapely         # geometry deps used by data_processing
```

## Running the demo (two laptops, same network)

**Laptop 1: API server**

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is required so Laptop 2 can reach the server. The server loads `models/model.pkl` at startup (override with `PATHSNAP_MODEL_PATH`). Find the LAN IP with `hostname -I` (Linux), `ipconfig getifaddr en0` (macOS) or `ipconfig` (Windows). Check it from Laptop 2:

```bash
curl http://<laptop1-ip>:8000/health
```

Interactive API docs: `http://<laptop1-ip>:8000/docs`.

**Laptop 2: evaluator UI**

```bash
streamlit run ui/app.py
```

In the sidebar, set **API Base URL** to `http://<laptop1-ip>:8000` and click **Check API**. Then **Fetch trips**, pick an evaluator trip, and **Run map matching**. The UI shows route F1, precision, recall, GPS points, route arcs, true-arc miss rate and skipped points, plus the map layers.

### API

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/health` | status, loaded model name, version |
| `GET` | `/trips` | evaluator trip IDs (and train IDs) |
| `POST` | `/predict` | body `{"trip_id": "00000030"}`; predicted arc sequence, GPS points, route F1 / precision / recall, true-arc miss rate |

## Reproducing the model results

Run from the project root, in order:

```bash
python data_processing/build_features.py    # -> results/features_train.parquet, features_studenttest.parquet
python models/compare_models.py             # 5-model GroupKFold comparison -> models_comparison.csv
python models/compare_ann.py                # adds the MLP -> models_comparison_all.csv
python models/train_final.py                # fits HistGB on all train trips -> models/model.pkl
python models/tune_optuna.py                # Bayesian tuning (Optuna TPE) -> models/model_tuned.pkl
python models/make_plots.py                 # report figures -> results/figures/
python models/confusion_matrix.py           # -> results/figures/confusion_matrix.png
python models/hmm_vs_classifier.py          # classifier-only vs classifier+HMM -> results/hmm_vs_classifier.csv
```

Unit tests:

```bash
python -m pytest tests/unit/ -v
```

## Results

**Candidate classifier**, 5-fold GroupKFold by trip on the 80 training trips. Models restricted to the course syllabus.

| Model | Macro-F1 | Top-1 accuracy | Log loss |
| --- | --- | --- | --- |
| **HistGradientBoosting** (selected) | **0.846** | 0.676 | 0.193 |
| Random Forest | 0.837 | 0.671 | 0.351 |
| MLP (ANN) | 0.837 | **0.689** | **0.192** |
| KNN | 0.800 | 0.629 | 0.665 |
| Gaussian Naive Bayes | 0.710 | 0.604 | 0.557 |
| Logistic Regression | 0.687 | 0.566 | 0.543 |

HistGB was selected for the best macro-F1, log loss level with the MLP and half that of Random Forest (the HMM uses these probabilities as emissions, so calibration matters), and the lowest tuning cost: an equivalent MLP search would have taken over 3 hours. On the held-out student-test trips it scores macro-F1 0.762, Top-1 0.869, log loss 0.160. Bayesian tuning (30 Optuna trials) moved CV macro-F1 from 0.846 to 0.851, so the defaults were already near-optimal.

**HMM / Viterbi** on the 10 student-test trips:

| | Route F1 | Route continuity |
| --- | --- | --- |
| Classifier only (argmax per point) | 0.745 | 0.799 |
| Classifier + HMM (Viterbi) | 0.678 | 0.959 |

The HMM makes the route connected and drivable, at a cost in arc-level F1 where the correct arcs are not connected in the candidate set. The deployed system uses the HMM. Route F1 on individual evaluator trips in the live demo falls between about 0.67 and 0.72, with recall above precision.

Full write-up, EDA and figures: see the project report.
