from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import io, math
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.metrics import r2_score

app = FastAPI()

# Mount static folder and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required_cols = {"PatientID", "Time", "Observed", "Dose"}
    if not required_cols.issubset(df.columns):
        return HTMLResponse(
            content="CSV must contain columns: PatientID, Time, Observed, Dose",
            status_code=400
        )

    results = []
    k_el = 0.15  # elimination constant (example)

    # Compute predicted values and residuals
    for _, row in df.iterrows():
        patient = row["PatientID"]
        dose = row["Dose"]
        t = row["Time"]

        pred = dose * math.exp(-k_el * t)

        # Avoid zero division for residual (%)
        pred_safe = pred if pred != 0 else 1e-6
        residual_pct = ((row["Observed"] - pred) / pred_safe * 100)

        results.append({
            "PatientID": patient,
            "Time": round(t, 2),
            "Observed": round(row["Observed"], 4),
            "Predicted": round(pred, 4),
            "Residual (%)": round(residual_pct, 2)
        })

    result_df = pd.DataFrame(results)

    # Compute patient-wise R² for plotting only
    patient_r2 = {}
    for pid, subdf in result_df.groupby("PatientID"):
        try:
            r2_val = r2_score(subdf["Observed"], subdf["Predicted"])
            r2_val = round(abs(r2_val), 2)
        except Exception:
            r2_val = 0.0
        patient_r2[pid] = r2_val

    # Save CSV for download (matching table)
    result_df.to_csv("static/prediction_results.csv", index=False)

    # Plot linear concentration vs time with patient R² in legend
    fig = go.Figure()
    for pid, subdf in result_df.groupby("PatientID"):
        fig.add_trace(go.Scatter(
            x=subdf["Time"], y=subdf["Observed"],
            mode="markers+lines",
            name=f"Observed P{pid} (R²={patient_r2[pid]})"
        ))
        fig.add_trace(go.Scatter(
            x=subdf["Time"], y=subdf["Predicted"],
            mode="lines",
            name=f"Predicted P{pid} (R²={patient_r2[pid]})"
        ))

    fig.update_layout(
        title="Concentration–Time Profile",
        xaxis_title="Time (h)",
        yaxis_title="Concentration (ng/mL)",
        template="plotly_white"
    )

    graph_html = plot(fig, output_type="div")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "graph_html": graph_html,
        "table_rows": results
    })


@app.get("/download")
async def download():
    return FileResponse(
        "static/prediction_results.csv",
        filename="prediction_results.csv"
    )


@app.get("/documentation", response_class=HTMLResponse)
async def documentation(request: Request):
    return templates.TemplateResponse("documentation.html", {"request": request})
