import asyncio
import nest_asyncio
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# --------------------------------------------------
# Router
# --------------------------------------------------
router = APIRouter(
    prefix="/kpi",
    tags=["KPI Agent"]
)

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
# Paths
# --------------------------------------------------
CLEANED_DATASET_DIR = BASE_DIR / "cleaned_datasets"
KPI_VECTOR_DIR = BASE_DIR / "kpi_rag_index"


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
# Column Semantic Detection
# --------------------------------------------------
def detect_column_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif series.nunique() < 20:
        return "categorical"
    elif pd.api.types.is_bool_dtype(series):
        return "boolean"
    else:
        return "text"


@tool("get_data_summary")
def load_data_summary(input_text: str = "") -> str:
    """Gets a detailed summary of the latest cleaned dataset. Use this FIRST."""
    df = load_latest_dataframe()
    if df is None:
        return (
            "ERROR: No cleaned dataset found. "
            "Please ensure at least one CSV file exists in the 'cleaned_datasets' directory."
        )

    try:
        column_info = {}
        for col in df.columns:
            column_info[col] = {
                "dtype": df[col].dtype.name,
                "semantic_type": detect_column_type(df[col]),
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
            f"=== COLUMN DETAILS (With Semantic Types) ===\n"
            f"{json.dumps(column_info, indent=2)}\n\n"
            f"=== SAMPLE DATA (First 5 Rows) ===\n"
            f"{sample_data_md}\n\n"
            f"Use this dataset to generate KPIs using single, dual and multi-column logic."
        )
        return summary

    except Exception as e:
        return f"ERROR: Failed to summarize dataset - {str(e)}"



"""
# --------------------------------------------------
# RAG Vector Store
# --------------------------------------------------
def build_or_load_kpi_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    if os.path.exists(KPI_VECTOR_DIR):
        return FAISS.load_local(
            KPI_VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Index already built; simply load existing vectorstore.
    # If the directory does not exist, raise an error.
    raise FileNotFoundError(f"KPI vector store not found at {KPI_VECTOR_DIR}. Please ensure the index is present.")


kpi_vectorstore = build_or_load_kpi_vectorstore()

# --------------------------------------------------
# TOOL 2: KPI RAG Retrieval
# --------------------------------------------------
def retrieve_kpi_knowledge(domain=None, columns=None, query=None):
    if domain or columns:
        domain = domain or ""
        if isinstance(columns, list):
            columns_text = ", ".join(columns)
        else:
            columns_text = str(columns)

        search_query = (
            f"Domain: {domain}. "
            f"Columns: {columns_text}. "
            f"Suggest KPI patterns using single, dual, and multi-column strategies."
        )
    elif query:
        search_query = query
    else:
        search_query = "General KPI patterns"

    results = kpi_vectorstore.similarity_search(search_query, k=5)
    context = "\n\n".join([doc.page_content for doc in results])

    return (
        "=== KPI KNOWLEDGE BASE CONTEXT (RAG) ===\n"
        "Use these examples as inspiration:\n\n"
        f"{context}"
    )


KPIKnowledgeTool = Tool(
    name="retrieve_kpi_knowledge",
    func=retrieve_kpi_knowledge,
    description="Retrieves relevant KPI patterns from the KPI knowledge base."
)
"""
# --------------------------------------------------
# RAG Vector Store (WITH METADATA)
# --------------------------------------------------
def build_or_load_kpi_vectorstore():
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    if os.path.exists(KPI_VECTOR_DIR):
        return FAISS.load_local(
            KPI_VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Index already built; load existing vectorstore.
    # If the directory does not exist, raise an informative error.
    raise FileNotFoundError(f"KPI vector store not found at {KPI_VECTOR_DIR}. Ensure the pre-built index is available.")

kpi_vectorstore = build_or_load_kpi_vectorstore()


@tool("retrieve_kpi_knowledge")
def retrieve_kpi_knowledge(domain: str = "", columns: Optional[List[str]] = None) -> str:
    """Retrieves domain-aligned KPI and non-KPI examples from the knowledge base for contrastive KPI reasoning."""

    column_text = ", ".join(columns) if columns else "unknown columns"

    # Strong semantic classification-style query
    search_query = (
        f"You are classifying dataset columns into KPI vs non-KPI.\n"
        f"Domain: {domain}\n"
        f"Columns present: {column_text}\n"
        f"Prefer numerical or aggregatable columns for KPIs.\n"
        f"Retrieve historical KPI vs non-KPI examples with similar column roles."
    )


    # --- Positive KPI examples ---
    positive_examples = kpi_vectorstore.similarity_search(
        search_query,
        k=3,
        filter={"is_kpi": 1, **({"domain": domain} if domain else {})}

    )

    # --- Negative NON-KPI examples ---
    negative_examples = kpi_vectorstore.similarity_search(
        search_query,
        k=2,
        filter={"is_kpi": 0, **({"domain": domain} if domain else {})}
    )

    # ---------- DOMAIN FALLBACK ----------
    if len(positive_examples) < 2:
        positive_examples = kpi_vectorstore.similarity_search(
            search_query,
            k=3,
            filter={"is_kpi": 1}
        )

    if len(negative_examples) < 1:
        negative_examples = kpi_vectorstore.similarity_search(
            search_query,
            k=2,
            filter={"is_kpi": 0}
        )


    positive_context = "\n\n".join(
        [doc.page_content for doc in positive_examples]
    )

    negative_context = "\n\n".join(
        [doc.page_content for doc in negative_examples]
    )

    return (
        "=== KPI KNOWLEDGE BASE (DOMAIN-MATCHED, CONTRASTIVE) ===\n\n"
        "POSITIVE KPI EXAMPLES (patterns to emulate):\n"
        f"{positive_context}\n\n"
        "NEGATIVE NON-KPI EXAMPLES (patterns to avoid):\n"
        f"{negative_context}\n\n"
        "Use the contrast between these examples to design strong, non-trivial KPIs."
    )


# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = LLM(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------------------------
# Agent + Task + Crew (UNCHANGED LOGIC)
# --------------------------------------------------
kpi_identification_agent = Agent(
    role="Strategic KPI Designer",
    goal=(
        "Design analytically strong, multi-dimensional KPIs using domain-aware "
        "patterns retrieved from a KPI knowledge base. "
        "Every KPI must be grounded in retrieved historical examples, not guesswork."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are a senior KPI architect.\n"
        "You do NOT invent KPIs from intuition alone.\n"
        "You ALWAYS consult a KPI knowledge base to understand:\n"
        "- What has historically been a KPI in similar domains\n"
        "- What should explicitly NOT be treated as a KPI\n\n"
        "You use retrieved examples as anchors and then adapt them intelligently "
        "to the current dataset."
    ),
    tools=[load_data_summary, retrieve_kpi_knowledge],
    llm=llm,
    allow_delegation=False
)

# --------------------------------------------------
# KPI TASK
# --------------------------------------------------
kpi_identification_task = Task(
    description=(
        "You are provided with:\n"
            "• Dataset Description (optional)\n"
            "• Dataset Domain (explicit — MUST be respected unless input is others)\n"
            "• Dataset Row Count (scale indicator)\n"
            "• RAG-based KPI Knowledge (via retrieve_kpi_knowledge)\n"

            "IMPORTANT RULES:\n"
            "• NEVER override or infer a different domain, please use semantics only to find related domains if needed\n"
            "• Use the provided domain as the primary filter\n"
            "• Dataset size MUST influence chart selection, make sure you take in consideration the number of rows before giving a chart\n"

        "CRITICAL RULES:\n"
        "1. The user-provided dataset domain is the PRIMARY domain.\n"
        "2. You MUST NOT override or ignore it unless its other or none.\n"
        "3. If an exact domain match is weak or missing in the KPI knowledge base:\n"
        "   - Use column names, dataset description, and table semantics\n"
        "   - Retrieve the closest RELATED domain patterns via the knowledge base\n\n"

        "You are REQUIRED to use the KPI knowledge base (RAG).\n"
        "All KPIs MUST be justified using retrieved positive and negative examples.\n\n"

        "PROCESS (MANDATORY ORDER):\n"
        "STEP 1: Call get_data_summary to understand columns and semantics.\n"
        "STEP 2: Use the USER-PROVIDED DOMAIN as-is.\n"
        "STEP 3: Call retrieve_kpi_knowledge with:\n"
        "        - domain = user dataset domain\n"
        "        - columns = column names from the dataset\n"
        "STEP 4: Analyze retrieved POSITIVE KPI patterns and NEGATIVE non-KPI patterns.\n"
        "STEP 5: Generate KPIs that:\n"
        "        - Align with positive patterns\n"
        "        - Explicitly avoid negative patterns\n\n"

        "STEP 3 is MANDATORY. You MUST call retrieve_kpi_knowledge before generating KPIs.\n"

        "MANDATORY KPI COMPOSITION:\n"
        "• At least 2 SINGLE-column KPIs (only if strongly justified)\n"
        "• At least 3 DUAL-column KPIs (ratios, dependencies, efficiencies)\n"
        "• At least 3 MULTI-column KPIs (3+ columns, composite business insights)\n\n"

        "AVOID:\n"
        "• Pure timestamps, IDs, or locations as KPIs\n"
        "• Simple counts or averages without business interpretation\n\n"

        "Each KPI MUST:\n"
        "• Reference dataset columns explicitly\n"
        "• Reflect a decision-making or operational insight\n"
        "• Be something a dashboard user would act upon\n\n"

        "Use Dataset Rows: {{dataset_rows}} to classify dataset size as:\n"
        "- Small (<5k)\n"
        "- Medium (5k–50k)\n"
        "- Large (>50k)\n"
        "and choose charts accordingly.\n\n"

        "CHART REQUIREMENTS:\n"
        "• Use a mix of basic and advanced charts based on its size:\n"
        "  Line, Bar, Heatmap, Funnel, Stacked Bar, Radar, Area, Boxplot, pie chart etc\n"
        "• Chart titles must be:\n"
        "  - Professional\n"
        "  - 4 to 7 words\n"
        "  - Business-meaning focused\n\n"

        "OUTPUT RULE:\n"
        "Output ONLY in the specified KPI format.\n"
        "Do NOT explain your process or mention internal tools."
    ),

    expected_output=(
        "KPI <n>:\n"
        "  Name: <Descriptive Professional KPI Name>\n"
        "  Columns: <Column names used>\n"
        "  Reasoning: <Business or operational justification, grounded in RAG examples>\n"
        "  Suggested Chart:\n"
        "    - Chart Type: <Chart type>\n"
        "    - Chart Title: <4 to 7 word professional title>\n\n"
    ),

    agent=kpi_identification_agent
)



crew = Crew(
    agents=[kpi_identification_agent],
    tasks=[kpi_identification_task],
    verbose=True
)


# --------------------------------------------------
# API SCHEMA
# --------------------------------------------------
class KPIRequest(BaseModel):
    dataset_description: Optional[str] = None
    dataset_domain: Optional[str] = None
    dataset_rows: Optional[int] = None


# --------------------------------------------------
# API ENDPOINT
# --------------------------------------------------
@router.post("/identify")
def identify_kpis(request: KPIRequest):
    inputs = {
        "user_dataset_description": request.dataset_description or "",
        "user_dataset_domain": request.dataset_domain or "",
        "dataset_rows": request.dataset_rows or 0
    }

    result = crew.kickoff(inputs=inputs)
    result_str = str(result)

    return {
        "status": "success",
        "kpi_output": result_str
    }
