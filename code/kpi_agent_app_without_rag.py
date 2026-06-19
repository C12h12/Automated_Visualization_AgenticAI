import asyncio
import nest_asyncio
import os
import json
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool


# --------------------------------------------------
# Fix event loop issues
# --------------------------------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Path to cleaned dataset directory
# --------------------------------------------------
CLEANED_DATASET_DIR = "cleaned_datasets"

# --------------------------------------------------
# Helper: Get latest CSV
# --------------------------------------------------
def get_latest_cleaned_csv() -> Optional[str]:
    try:
        files = [
            os.path.join(CLEANED_DATASET_DIR, f)
            for f in os.listdir(CLEANED_DATASET_DIR)
            if f.endswith(".csv")
        ]
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    except Exception:
        return None


def load_latest_dataframe() -> Optional[pd.DataFrame]:
    latest_file = get_latest_cleaned_csv()
    if latest_file is None:
        return None
    try:
        return pd.read_csv(latest_file)
    except Exception:
        return None


# --------------------------------------------------
@tool("get_data_summary")
def load_data_summary(input_text: str = "") -> str:
    """Gets a summary of the latest cleaned dataset CSV from the cleaned_dataset directory. Includes row count, column info, null values, unique values, and sample rows. Use this FIRST before identifying KPIs."""
    df = load_latest_dataframe()
    if df is None:
        return (
            "ERROR: No cleaned dataset found. "
            "Please ensure at least one CSV file exists in the 'cleaned_dataset' directory."
        )

    try:
        column_info = {}
        for col in df.columns:
            column_info[col] = {
                "dtype": df[col].dtype.name,
                "null_count": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique())
            }

        try:
            sample_data_md = df.head(5).to_markdown(index=False)
        except:
            sample_data_md = df.head(5).to_string(index=False)

        summary = (
            f"=== DATASET SUMMARY (Latest Cleaned File) ===\n\n"
            f"File Used: {get_latest_cleaned_csv()}\n"
            f"Total Rows: {len(df)}\n"
            f"Total Columns: {len(df.columns)}\n\n"
            f"=== COLUMN DETAILS ===\n"
            f"{json.dumps(column_info, indent=2)}\n\n"
            f"=== SAMPLE DATA (First 5 Rows) ===\n"
            f"{sample_data_md}\n\n"
            f"You can now identify KPIs based on this dataset."
        )
        return summary

    except Exception as e:
        return f"ERROR: Failed to summarize dataset - {str(e)}"


# --------------------------------------------------
# OpenAI LLM
# --------------------------------------------------
llm = LLM(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

# --------------------------------------------------
# KPI IDENTIFICATION AGENT
# --------------------------------------------------
kpi_identification_agent = Agent(
    role="Expert KPI Analyst",
    goal=(
        "Analyze the dataset using the get_data_summary tool and identify 5-8 key performance indicators (KPIs) "
        "that would be most valuable for business insights and visualization."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are a senior data analyst with 10+ years of experience in identifying business KPIs. "
        "Your expertise lies in understanding data structures and determining which metrics matter most. "
        "\n\nYour workflow:\n"
        "1. FIRST: Use the 'get_data_summary' tool to understand the dataset\n"
        "2. ANALYZE: Review column types, null values, and sample data\n"
        "3. IDENTIFY: Determine 5-8 meaningful KPIs based on the data\n"
        "4. DOCUMENT: For each KPI, specify:\n"
        "   - KPI Name\n"
        "   - Column(s) used\n"
        "   - Data type\n"
        "   - Business reasoning\n"
        "   - Suggested visualization type\n"
        "\nIMPORTANT: You MUST use the get_data_summary tool as your first action."
    ),
    tools=[load_data_summary],
    llm=llm,
    allow_delegation=True
)

# --------------------------------------------------
# KPI TASK
# --------------------------------------------------
kpi_identification_task = Task(
    description=(
        "STEP 1: Use the 'get_data_summary' tool to retrieve dataset information.\n"
        "STEP 2: Analyze the dataset structure, columns, data types, and sample values.\n"
        "STEP 3: Identify 5-8 meaningful KPIs that would provide valuable business insights.\n"
        "STEP 4: For EACH KPI you identify, provide:\n"
        "   a) KPI Name\n"
        "   b) Column(s)\n"
        "   c) Data type\n"
        "   d) Business reasoning\n"
        "   e) Suggested visualization type\n"
    ),
    expected_output=(
        "A structured list of 5–8 KPIs in the following format:\n\n"
        "KPI 1:\n"
        "  Name: <KPI Name>\n"
        "  Columns: <Column Names>\n"
        "  Data Type: <Data Type>\n"
        "  Reasoning: <Why this KPI matters>\n"
        "  Suggested Chart: <Chart Type>\n\n"
        "[Repeat for all KPIs]"
    ),
    agent=kpi_identification_agent
)


# --------------------------------------------------
# CREW
# --------------------------------------------------
crew = Crew(
    agents=[kpi_identification_agent],
    tasks=[kpi_identification_task],
    verbose=True
)

# --------------------------------------------------
# FASTAPI
# --------------------------------------------------
app = FastAPI(title="KPI Identification Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class KPIRequest(BaseModel):
    run: bool = True


@app.post("/identify-kpis")
def identify_kpis(request: KPIRequest):
    try:
        result = crew.kickoff()

        # 🔥 Convert result safely to string
        result_str = str(result)

        return {
            "status": "success",
            "kpi_output": result_str
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

