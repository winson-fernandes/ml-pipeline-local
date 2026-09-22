# Heart Disease ML Pipeline

A complete, end to end machine learning pipeline that runs entirely on your
own machine, with continuous integration and continuous deployment through
GitHub Actions.

The pipeline downloads the UCI Heart Disease dataset, cleans it, engineers
features, trains a PyTorch neural network, evaluates it, deploys it to a
local model server, and monitors that server's predictions for drift. Every
stage is a plain Python script that reads its input from a local folder and
writes its output to another local folder. You can run the whole thing,
inspect any intermediate file, or rerun a single stage, all without any
external service.

## Prerequisites

You need the following installed before you start. None of them require a
paid account or a GPU.

- **Git**, to clone this repository.
- **Python 3.10, 3.11, or 3.12**. Run `python3 --version` (or `python
  --version` on Windows) to check what you have. If you use conda instead,
  it can create an environment with the right Python version for you, so a
  separate system-wide Python install is not required in that case.
- **pip**, which ships with Python and with every conda environment.
- **make**, to run the shortcuts in the Makefile. Linux and macOS have it
  by default. On Windows, install it through WSL or Git Bash, or skip it
  and run the underlying `python -m src.<module>` commands directly,
  listed under "Running each stage individually" below.
- An **internet connection**, needed once, to download the dataset.

A GPU is optional. Every stage runs fine on CPU.

## Setup

First clone the repository.

```bash
git clone <this-repository-url>
cd ml-pipeline-local
```

Then create an isolated Python environment using either `venv` or `conda`.
Both give the same result. Use whichever you already have.

### Option A: venv

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

On Windows, activate the environment with `venv\Scripts\activate` instead
of `source venv/bin/activate`.

On Debian or Ubuntu, creating a virtual environment can fail with
"ensurepip is not available." If that happens, install the matching
`python3.X-venv` package first, for example `sudo apt install
python3.12-venv`, then retry the `python3 -m venv venv` command.

### Option B: conda

```bash
conda create -n ml-pipeline-local python=3.12
conda activate ml-pipeline-local

pip install -r requirements.txt
```

This project's dependencies are only published on PyPI, not on any conda
channel, so `pip install -r requirements.txt` is still the right command
here too. Conda is only being used to create and manage the Python
environment itself, the same role `venv` plays in Option A.

Any of 3.10, 3.11, or 3.12 works in place of 3.12 in that command.

### After either option

You must activate this environment in every new terminal window before
running any command in this README. With venv that means running the
`source venv/bin/activate` line again. With conda it means running `conda
activate ml-pipeline-local` again. If a command fails with "module not
found," activating the environment in that terminal is usually the fix.

## Getting started

Run the full pipeline once, end to end:

```bash
make pipeline
```

This downloads the dataset, preprocesses it, trains a model, evaluates it,
and deploys it. It takes a few minutes on CPU.

Then start the local model server and ask it for a prediction:

```bash
make serve      # keep this running in its own terminal
make predict    # in a second terminal, with the virtual environment activated
```

## Pipeline stages

1. **Download** (`src/data/data_download.py`) downloads the UCI Heart
   Disease dataset to `data/raw/`.
2. **Preprocess** (`src/data/data_preprocess.py`) handles missing values,
   engineers 6 additional features (13 raw features become 19), splits the
   data into train, validation, and test sets (70/15/15), and normalizes it
   with a `StandardScaler`. It writes `data/processed/*.npz` and
   `models/scalers/*.pkl`.
3. **Train** (`src/model/model_train.py`) trains `HeartDiseaseNN`, a small
   PyTorch feed forward network (19 to 128 to 64 to 32 to 2) with batch
   normalization, dropout, early stopping, and a learning rate scheduler. It
   writes `models/output/run_<timestamp>/model.pth`, along with the
   training history and metrics.
4. **Evaluate** (`src/evaluation/model_evaluate.py`) scores the held out
   test set (accuracy, precision, recall, F1, AUC) and saves a confusion
   matrix and ROC curve to `evaluation/`.
5. **Deploy** (`src/deployment/model_deploy.py`) promotes a trained model
   into `models/registry/latest/`, the folder the local server always
   serves from, and runs a self test prediction.
6. **Serve** (`src/serving/serve.py`) is a small FastAPI process exposing
   `GET /health` and `POST /predict` as a plain local HTTP server.
7. **Monitor** (`src/monitoring/monitor_pipeline.py`) reads the local
   server's prediction log and flags feature drift by comparing recent
   request features against the training time feature statistics.

The model architecture (`src/common/model_def.py`) and the raw to feature
transform (`src/common/features.py`) are each defined once and shared by
every stage that needs them. Training and serving can never silently drift
apart.

## Running each stage individually

```bash
make download
make preprocess
make train
make evaluate
make deploy
make serve      # keep running in its own terminal
make predict    # in a second terminal
make monitor
```

Without `make`, run the equivalent command directly, for example `python -m
src.data.data_download` instead of `make download`. Every target in the
Makefile maps to one `python -m src.<module>` command.

`python -m src.pipeline --help` shows options like `--epochs`,
`--skip-download`, `--skip-preprocess`, and `--no-deploy`.

## Iterating without redoing earlier stages

Each stage script only touches its own inputs and outputs, and always reads
whatever is currently newest from disk. You never have to redo data cleanup
just to try a new model idea.

```bash
make train      # trains against the newest data/processed/*.npz
make evaluate   # evaluates the newest models/output/run_*/
make deploy     # promotes the newest run to models/registry/latest/
```

Repeat this train, evaluate, deploy loop as many times as you like. Each
`train` call drops a fresh `models/output/run_<timestamp>/`. `evaluate` and
`deploy` always pick up the most recent one. Only rerun `make download` or
`make preprocess` when the raw data or the feature engineering itself
changes.

Every stage also takes an explicit path override, so you can deploy or
evaluate a specific older run instead of the newest one:

```bash
python -m src.evaluation.model_evaluate --model-dir models/output/run_20260101_120000
```

`model_deploy.py` accepts the equivalent `--model-dir` flag.

## Predicting

Once the server is running (`make serve`), send it a raw feature vector.
It expects the 13 clinical values plus the 6 engineered ones, in the order
defined by `FEATURE_COLUMNS` in `src/common/features.py`.

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 9.135, 3.698, 2.381, 0, 1, 0]}'
```

Or use the CLI client, which builds that payload for you:

```bash
python -m src.inference.predict --features '[63,1,3,145,233,1,0,150,0,2.3,0,0,1,9.135,3.698,2.381,0,1,0]'
```

Or on UI: once "make serve" or "python -m src.serving.serve" is executed, go to the option under /predict and set the "Parameters" option and set the features to "

{
  "features": [63,1,3,145,233,1,0,150,0,2.3,0,0,1,9.135,3.698,2.381,0,1,0]
}
and run "Execute"

The server normalizes this raw feature vector with the model's fitted
`StandardScaler` before running it through the network. Send raw clinical
values here, not pre-scaled ones.

## Testing and linting

```bash
pip install -r requirements-dev.txt

pytest          # runs the unit test suite in tests/
ruff check .    # lints the codebase
```

The unit tests cover the feature engineering transform, the model
architecture, the inference handlers, the path utilities, and the drift
detection logic. They run in well under a second and need no network
access, no trained model, and no downloaded data.

## Continuous integration and continuous deployment

`.github/workflows/ml-pipeline.yml` runs on every push and pull request to
`main`, and on demand from the Actions tab, with an `epochs` input. It has
six jobs.

1. **Lint and test** runs `ruff` and the unit test suite. Every later job
   depends on this one passing.
2. **Download** fetches the dataset.
3. **Preprocess** cleans it and engineers features.
4. **Train** trains the model.
5. **Evaluate** scores it on the held out test set.
6. **Deploy** promotes the model into the registry, starts the local model
   server, and sends it one real prediction request to confirm it responds
   correctly before the run is considered successful.

Each job starts on a clean runner with nothing shared between them.
Artifacts pass from job to job using `actions/upload-artifact` and
`actions/download-artifact`. Download any stage's output, such as the
trained model or the evaluation plots, from the run's Artifacts section in
the GitHub Actions UI.

The download and preprocess jobs each restore from an `actions/cache` entry
keyed on a hash of the code that produces their output, and, for
preprocess, the raw CSV's own hash. If that hash has not changed since the
last run, the download or preprocess step is skipped and the cached files
are reused. A push that only changes `model_train.py` or the
hyperparameters goes straight to training instead of redoing data work. The
train, evaluate, and deploy jobs always run. To force a fresh download and
preprocess, bump `CACHE_VERSION` at the top of the workflow file.

## Web interface

`frontend-app/` is a prediction UI with a single patient form and a CSV
batch upload, calling the local model server. See `frontend-app/README.md`
for how to run it. It needs the model server from `make serve` running
first.

## Project structure

```
ml-pipeline-local/
├── src/
│   ├── common/        model definition, feature engineering, and paths
│   ├── data/           download and preprocess
│   ├── model/          training
│   ├── evaluation/      test-set metrics and plots
│   ├── deployment/      promotes a model into the local registry
│   ├── serving/         local FastAPI model server
│   ├── inference/       inference handlers and the CLI predict client
│   ├── monitoring/      drift check over the local prediction log
│   └── pipeline.py      runs every stage end to end
├── tests/               unit tests
├── data/raw/, data/processed/
├── models/output/, models/scalers/, models/registry/latest/
├── evaluation/
├── monitoring/
├── frontend-app/        web interface (FastAPI and Flask)
├── Makefile
├── requirements.txt
└── requirements-dev.txt
```

## License

MIT License
