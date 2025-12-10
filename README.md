## How to run this project

1️⃣ Clone the repo & set up environment
git clone https://github.com/<YOUR_USERNAME>/mlops-project-1-ml-pipeline.git
cd mlops-project-1-ml-pipeline

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

2️⃣ Train the model
python -m src.ml_pipeline.train


This will:

load dataset

train the ML pipeline

save logs → logs/training.log

save metrics → logs/metrics.json

save model → models/model.joblib

3️⃣ Run inference from the script
python -m src.ml_pipeline.inference

4️⃣ Start the FastAPI server
uvicorn src.api.app:app --reload

5️⃣ Health check

Open in your browser:

http://127.0.0.1:8000/health

6️⃣ Make a prediction (example with 30 zeros)
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'

  This project is a modular ML pipeline with:

configuration-driven training

metrics + logs output

saved model artifacts

script-based inference

FastAPI inference service

The repository will be updated as the project develops.