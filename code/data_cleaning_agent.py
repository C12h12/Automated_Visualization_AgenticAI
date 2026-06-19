
from fastapi import APIRouter, UploadFile, File, HTTPException, Response, status
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import io
from collections import defaultdict
import uuid
import os
from datetime import datetime

# =================================================
# Router (IMPORTANT CHANGE)
# =================================================
router = APIRouter(prefix="/cleaning", tags=["Data Cleaning Agent"])

# =================================================
# Storage
# =================================================
DATASETS = {}  # in-memory dataset store

UPLOAD_DIR = "uploaded_datasets"
CLEANED_DATA_DIR = "cleaned_datasets"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

# =================================================
# Schemas
# =================================================
class CleaningAction(BaseModel):
    column: str
    step_id: str
    recommended_action: str
    problem: Optional[str]
    action_reason: Optional[str]
    risk_if_ignored: Optional[str]

class CleaningRequest(BaseModel):
    dataset_id: str
    actions: List[CleaningAction]
    dry_run: bool = False

# =================================================
# Step execution order
# =================================================
STEP_ORDER = [
    "remove_duplicates",
    "handle_missing_values",
    "correct_data_types",
    "unify_date_formats",
    "validate_numeric_ranges",
]

# =================================================
# Cleaning functions
# =================================================
def remove_duplicates(df):
    return df.drop_duplicates()

def drop_rows(df, column):
    return df.dropna(subset=[column])

def impute_median(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        df.loc[:, column] = df[column].fillna(df[column].median())
    return df

def impute_mode(df, column):
    mode = df[column].mode(dropna=True)
    if not mode.empty:
        df.loc[:, column] = df[column].fillna(mode[0])
    return df

def cast_to_float(df, column):
    df.loc[:, column] = pd.to_numeric(df[column], errors="coerce")
    return df

def standardize_date(df, column):
    df.loc[:, column] = pd.to_datetime(df[column], errors="coerce")
    return df

def cap_values(df, column):
    df.loc[:, column] = pd.to_numeric(df[column], errors="coerce")
    df.loc[:, column] = df[column].clip(lower=0)
    return df

# =================================================
# Action registries
# =================================================
COLUMN_ACTIONS = {
    "DROP_ROWS": drop_rows,
    "IMPUTE_MEDIAN": impute_median,
    "IMPUTE_MODE": impute_mode,
    "CAST_TO_FLOAT": cast_to_float,
    "STANDARDIZE_FORMAT": standardize_date,
    "CAP_VALUES": cap_values,
}

DATASET_ACTIONS = {
    "REMOVE_DUPLICATES": remove_duplicates
}

# =================================================
# JSON safety helpers
# =================================================
def make_json_safe(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return None if np.isinf(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value

def sanitize_obj(obj):
    if isinstance(obj, dict):
        return {k: sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_obj(v) for v in obj]
    return make_json_safe(obj)

# =================================================
# Core cleaning engine
# =================================================
def execute_cleaning_plan(df, actions, dry_run=False):
    logs = []
    run_id = str(uuid.uuid4())

    grouped = defaultdict(list)
    for action in actions:
        grouped[action.step_id].append(action)

    for step in STEP_ORDER:
        for action in grouped.get(step, []):
            log = {
                "run_id": run_id,
                "step": step,
                "column": action.column,
                "action": action.recommended_action,
                "executed": False,
                "reason": action.action_reason,
                "risk_if_ignored": action.risk_if_ignored
            }

            if action.column == "__dataset__":
                fn = DATASET_ACTIONS.get(action.recommended_action)
                if fn and not dry_run:
                    df = fn(df)
                    log["executed"] = True
                logs.append(log)
                continue

            if action.column not in df.columns:
                logs.append(log)
                continue

            fn = COLUMN_ACTIONS.get(action.recommended_action)
            if not fn:
                logs.append(log)
                continue

            if not dry_run:
                df = fn(df, action.column)
                log["executed"] = True

            logs.append(log)

    return df, logs, run_id

# =================================================
# API: Upload dataset
# =================================================
@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    dataset_id = str(uuid.uuid4())

    file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    df.to_csv(file_path, index=False)

    return {
        "dataset_id": dataset_id,
        "rows": len(df),
        "columns": list(df.columns)
    }

# =================================================
# API: Get metadata
# =================================================
@router.get("/metadata/{dataset_id}")
def get_metadata(dataset_id: str):
    file_path = f"{UPLOAD_DIR}/{dataset_id}.csv"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(file_path)

    metadata = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "inferred_columns": {}
    }

    for col in df.columns:
        series = df[col]
        missing = series.isna().sum()

        col_meta = {
            "missing_percentage": round((missing / len(df)) * 100, 2),
            "dtype": str(series.dtype),
            "unique_values": int(series.nunique(dropna=True))
        }

        metadata["inferred_columns"][col] = sanitize_obj(col_meta)

    return metadata

# =================================================
# API: Execute cleaning
# =================================================
@router.post("/execute-cleaning", status_code=status.HTTP_204_NO_CONTENT)
def execute_cleaning(request: CleaningRequest):
    file_path = f"{UPLOAD_DIR}/{request.dataset_id}.csv"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")

    df = pd.read_csv(file_path)

    cleaned_df, logs, run_id = execute_cleaning_plan(
        df,
        request.actions,
        request.dry_run
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{request.dataset_id}_{run_id}_{timestamp}.csv"
    cleaned_df.to_csv(os.path.join(CLEANED_DATA_DIR, filename), index=False)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
