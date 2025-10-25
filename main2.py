from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from starlette.requests import Request

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import plotly.graph_objs as go
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class PopulationODEFunc(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def forward(self, t, y):
        k = F.softplus(self.net(y))
        return -k * y


def compute_metrics(t_grid, y_pred, dose_val=None):
    c = np.maximum(y_pred, 0.0)
    auc = float(np.trapz(c, t_grid))
    cmax = float(np.max(c))
    tmax = float(t_grid[np.argmax(c)])
    
    dt = np.diff(t_grid)
    dc = np.diff(c)
    tiny = 1e-8
    with np.errstate(divide='ignore', invalid='ignore'):
        k_inst = - (dc / dt) / (c[:-1] + tiny)
    k_valid = k_inst[np.isfinite(k_inst) & (k_inst > 0)]
    half_life = np.log(2)/np.median(k_valid) if k_valid.size>0 else None
    clearance = round(float(dose_val/auc), 2) if (dose_val is not None and auc>0) else None
    
    return round(auc,2), round(cmax,2), round(tmax,2), (round(half_life,2) if half_life else None), clearance


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if not all(col in df.columns for col in ["PatientID", "Time", "Observed", "Dose"]):
        return HTMLResponse("<h3>CSV must contain columns: PatientID, Time, Observed, Dose</h3>")

    patient_ids = df["PatientID"].unique()
    patient_data = []
    for pid in patient_ids:
        sub = df[df["PatientID"] == pid]
        patient_data.append({
            "PatientID": pid,
            "Time": sub["Time"].values,
            "Observed": sub["Observed"].values,
            "Dose": float(sub["Dose"].iloc[0])
        })

    # Train Neural ODE (population)
    func = PopulationODEFunc(hidden_dim=32).to(DEVICE)
    optimizer = torch.optim.Adam(func.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    y0_list = [torch.tensor([pat["Observed"][0]], dtype=torch.float32, device=DEVICE) for pat in patient_data]
    t_list = [torch.tensor(pat["Time"], dtype=torch.float32, device=DEVICE) for pat in patient_data]
    y_all = torch.cat([torch.tensor(pat["Observed"], dtype=torch.float32, device=DEVICE).unsqueeze(1) for pat in patient_data])

    best_loss = float("inf")
    wait = 0
    patience = 60
    epochs = 300

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred_list = []
        for y0, t_tensor in zip(y0_list, t_list):
            y_pred = odeint(func, y0.unsqueeze(0), t_tensor).squeeze(1)
            y_pred_list.append(y_pred)
        y_pred_all = torch.cat(y_pred_list, dim=0).unsqueeze(1)
        loss = loss_fn(y_pred_all, y_all)
        loss.backward()
        optimizer.step()

        curr_loss = loss.item()
        if curr_loss < best_loss - 1e-8:
            best_loss = curr_loss
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    # Predictions + Metrics
    table_rows = []
    fig = go.Figure()
    for pat in patient_data:
        t_grid = np.linspace(min(pat["Time"]), max(pat["Time"]), 200)
        y0 = torch.tensor([pat["Observed"][0]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            y_pred = odeint(func, y0.unsqueeze(0), torch.tensor(t_grid, dtype=torch.float32, device=DEVICE)).squeeze(1).cpu().numpy()
        auc, cmax, tmax, half_life, clearance = compute_metrics(t_grid, y_pred, dose_val=pat["Dose"])
        for t_val, obs_val, pred_val in zip(pat["Time"], pat["Observed"], y_pred[::len(y_pred)//len(pat["Time"])]):
            table_rows.append({
                "PatientID": pat["PatientID"],
                "Time": t_val,
                "Observed": round(obs_val,2),
                "Predicted": round(float(pred_val),2)
            })

        fig.add_trace(go.Scatter(x=pat["Time"], y=pat["Observed"], mode='markers', name=f"{pat['PatientID']} Observed"))
        fig.add_trace(go.Scatter(x=t_grid, y=y_pred, mode='lines', name=f"{pat['PatientID']} Predicted"))

    csv_file = "predictions.csv"
    pd.DataFrame(table_rows).to_csv(csv_file, index=False)

    fig.update_layout(title="Neural ODE PK Predictions", xaxis_title="Time", yaxis_title="Concentration")
    graph_html = fig.to_html(full_html=False)

    return templates.TemplateResponse("index.html", {"request": request, "graph_html": graph_html, "csv_file": csv_file, "table_rows": table_rows})
    

@app.get("/download")
async def download_csv():
    return FileResponse("predictions.csv", media_type='text/csv', filename="predictions.csv")
