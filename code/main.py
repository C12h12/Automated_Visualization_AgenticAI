
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
import uuid  # for generating dataset_id

from llm_review_agent import router as llm_review_router
from data_cleaning_agent import router as cleaning_router
from kpi_agent_app import router as kpi_router
from chart_generator import router as chart_router

app = FastAPI()

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev only
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include LLM router
app.include_router(llm_review_router)
#Data Cleaning Agent Router 
app.include_router(cleaning_router)
#KPI Agent Route
app.include_router(kpi_router)
#Chart Generator
app.include_router(chart_router)

# ------------------------------
# Home page -> final.html
# ------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("static/final.html", "r", encoding="utf-8") as f:
        return f.read()

# ------------------------------
# Upload page -> index.html
# ------------------------------
@app.get("/index", response_class=HTMLResponse)
def serve_index():
    with open("html/index.html", "r", encoding="utf-8") as f:
        return f.read()
    
@app.get("/metadata", response_class=HTMLResponse)
def serve_metadata():
    with open("html/metadata.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/review", response_class=HTMLResponse)
def serve_review():
    with open("html/review.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/kpi", response_class=HTMLResponse)
def serve_kpi_ui():
    with open("html/kpi_ui.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    with open("html/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

# ------------------------------
# Upload dataset endpoint
# ------------------------------
# @app.post("/upload-dataset")
# async def upload_dataset(file: UploadFile = File(...)):
#     """
#     Receives CSV or Excel file from frontend,
#     returns dataset_id and basic confirmation.
#     """
#     contents = await file.read()
#     filename = file.filename.lower()

#     try:
#         if filename.endswith(".csv"):
#             df = pd.read_csv(BytesIO(contents))
#         elif filename.endswith((".xlsx", ".xls")):
#             df = pd.read_excel(BytesIO(contents))
#         else:
#             return JSONResponse({
#                 "message": f"File '{file.filename}' uploaded successfully ✅",
#                 "dataset_id": str(uuid.uuid4())  # dummy ID
#             })
#     except Exception as e:
#         return JSONResponse({"message": f"Error reading file: {str(e)}"}, status_code=400)

#     # Here you can run your agents for cleaning/KPI/Charts if needed
#     # Example placeholders:
#     # cleaned_df = DataCleaningAgent(df).run()
#     # kpis = KPIIdentificationAgent(cleaned_df).run()
#     # charts = VisualizationAgent(cleaned_df).generate_charts()

#     # For now, we just return a dataset_id to frontend
#     dataset_id = str(uuid.uuid4())

#     return JSONResponse({
#         "message": f"Dataset '{file.filename}' uploaded successfully ✅",
#         "dataset_id": dataset_id
#     })
