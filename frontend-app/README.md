# Heart Disease Prediction Web Application

A web interface for heart disease prediction. It calls a model server that
runs entirely on your own machine.

## Features

**Single Patient Prediction**
- Interactive form with all 13 clinical features
- Real-time prediction results
- Risk level assessment

**Batch/CSV Prediction**
- Upload CSV files with multiple patients
- Bulk prediction processing, downloadable results

**RESTful API**
- `/predict` - Single prediction
- `/predict/batch` - Multiple predictions
- `/predict/csv` - CSV file upload

## Prerequisites

This app calls the local model server from the repo root. Before starting it,
train and deploy a model once:

```bash
cd ..
make pipeline
python -m src.serving.serve
```

`make pipeline` runs download, preprocess, train, evaluate, and deploy in
order. The second command starts the model server on
`http://localhost:8080`.

## Quick Start

### Option 1: Docker Compose

```bash
docker-compose up -d
# Web UI: http://localhost:5000
# API:    http://localhost:8000  (docs at /docs)
```

This starts three containers: the local model server (port 8080), the
prediction API (port 8000), and the web UI (port 5000).

### Option 2: Run directly

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Start the model server (from the repo root, in another terminal):**
```bash
cd ..
python -m src.serving.serve
```

**3. Start the API:**
```bash
cd backend
export MODEL_SERVER_URL=http://localhost:8080
python api.py
```

**4. Start the web server (new terminal):**
```bash
cd backend
export API_URL=http://localhost:8000
python app.py
```

**5. Access:**
- Web UI: http://localhost:5000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

```bash
curl -X POST http://localhost:8000/predict/csv -F "file=@patients.csv"
```

## CSV Format

```
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
63,1,3,145,233,1,0,150,0,2.3,0,0,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2
```

## Feature Descriptions

| Feature | Description | Values |
|---------|-------------|--------|
| age | Age in years | 1-120 |
| sex | Sex | 0=Female, 1=Male |
| cp | Chest pain type | 0-3 |
| trestbps | Resting blood pressure (mm Hg) | 50-250 |
| chol | Serum cholesterol (mg/dl) | 100-600 |
| fbs | Fasting blood sugar > 120 mg/dl | 0=No, 1=Yes |
| restecg | Resting ECG results | 0-2 |
| thalach | Maximum heart rate | 50-250 |
| exang | Exercise induced angina | 0=No, 1=Yes |
| oldpeak | ST depression | 0-10 |
| slope | Slope of peak exercise ST | 0-2 |
| ca | Major vessels (0-3) | 0-3 |
| thal | Thalassemia | 1-3 |

## Project Structure

```
frontend-app/
├── backend/
│   ├── api.py              # FastAPI backend (calls the local model server)
│   └── app.py               # Flask web server
├── frontend/
│   ├── templates/
│   └── static/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.web
├── requirements.txt
└── README.md
```

## Configuration

- `MODEL_SERVER_URL` (backend/api.py) - local model server URL, default `http://localhost:8080`
- `API_URL` (backend/app.py) - API server URL, default `http://localhost:8000`

## Troubleshooting

**"Model server unavailable" / 503 on `/health`**
The local model server isn't running, or no model has been deployed yet. From
the repo root: `make pipeline` then `python -m src.serving.serve`.

**Port already in use**
Change ports in `docker-compose.yml`, or when running directly, edit the
`app.run(...)` call at the bottom of `backend/app.py` (it always binds to
port 5000; there is no `--port` command-line flag).

## License

MIT License
