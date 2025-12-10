## How to run this project

### 1. Clone the repo & set up environment

```bash
git clone https://github.com/<YOUR_USERNAME>/mlops-project-1-ml-pipeline.git
cd mlops-project-1-ml-pipeline

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

2. Train the model
python -m src.ml_pipeline.train

Run inference from the script
python -m src.ml_pipeline.inference

Start the FastAPI server
uvicorn src.api.app:app --reload

Health check:
GET http://127.0.0.1:8000/health

Make a prediction (example):
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'



---

This repository will be updated as the project develops.
