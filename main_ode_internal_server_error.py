from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import torch
from torchdiffeq import odeint
import os
import plotly.graph_objects as go

app = FastAPI()

# Mount folders
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
OUTPUT_FILE = os.path.join(UPLOAD_FOLDER, "predictions.csv")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------
# Neural ODE PK Model
# ------------------------
class NeuralODEPK(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, t, y):
        return self.net(y)

model = NeuralODEPK()

def predict_concentration(dose, times):
    y0 = torch.tensor([[0.0, dose]], dtype=torch.float32)
    t = torch.tensor(times, dtype=torch.float32)
    y_pred = odeint(model, y0, t)
    conc = y_pred[:, 0, 0].detach().numpy()
    conc = np.maximum(conc, 0)           # avoid negatives
    conc = np.round(conc, 2)
    return conc

# ------------------------
# PK Metric Calculations
# ------------------------
def compute_metrics(times, conc):
    auc = np.round(np.trapz(conc, times), 2)
    cmax = np.round(np.max(conc), 2)
    tmax = np.round(times[np.argmax(conc)], 2)
    return auc, cmax, tmax

# ------------------------
# Routes
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "graph_html": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile):
    df = pd.read_csv(file.file)
    required_cols = ["PatientID", "Time", "Observed", "Dose"]
    if not all(col in df.columns for col in required_cols):
        return HTMLResponse(content="CSV must contain columns: PatientID, Time, Observed, Dose", status_code=400)

    results = []
    fig = go.Figure()

    for pid, group in df.groupby("PatientID"):
        times = group["Time"].values
        dose = group["Dose"].iloc[0]
        observed = group["Observed"].values
        conc_pred = predict_concentration(dose, times)

        # Compute % residuals
        residual_pct = np.round(((observed - conc_pred) / (conc_pred + 1e-6)) * 100, 2)

        # Save results
        for i in range(len(times)):
            results.append({
                "PatientID": pid,
                "Time": times[i],
                "Observed": observed[i],
                "Predicted": conc_pred[i],
                "Residual (%)": residual_pct[i]
            })

        # Plot linear concentration–time profile
        fig.add_trace(go.Scatter(
            x=times, y=conc_pred,
            mode="lines+markers", name=f"{pid} Predicted"
        ))
        fig.add_trace(go.Scatter(
            x=times, y=observed,
            mode="markers", name=f"{pid} Observed"
        ))

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_FILE, index=False)

    graph_html = fig.to_html(full_html=False)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "graph_html": graph_html,
        "table_rows": results
    })

@app.get("/download")
async def download_csv():
    return FileResponse(OUTPUT_FILE, filename="predictions.csv")

@app.get("/documentation", response_class=HTMLResponse)
async def documentation(request: Request):
    return templates.TemplateResponse("documentation.html", {"request": request})
