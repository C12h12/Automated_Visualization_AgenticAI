
from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# APIRouter (IMPORTANT CHANGE)
# -------------------------------
router = APIRouter(
    prefix="/llm-agent",
    tags=["LLM Review Agent"]
)

# -------------------------------
# Request schema
# -------------------------------
class ReviewRequest(BaseModel):
    dataset_metadata: dict
    sample_rows: list
    user_context: dict

# -------------------------------
# LLM Prompts
# -------------------------------
SYSTEM_PROMPT = """
You are a senior data quality and data cleaning advisor.

IMPORTANT RULES:
- Your response MUST be valid JSON
- Do NOT include explanations, markdown, or text outside JSON
- Do NOT wrap JSON in ``` or quotes
- The first character of your response MUST be {

If you cannot comply, return an empty JSON object {}.
"""
USER_PROMPT_TEMPLATE = """
You are a senior data quality analyst.

Task:
- Analyze the dataset column-wise using metadata and sample rows.

Dataset metadata:
{metadata}

Sample rows:
{rows}

Return STRICT JSON ONLY in this format:

{{
  "issues": [
    {{
      "column": "column_name",
      "step_id": "one_of_the_allowed_steps",
      "recommended_action": "one_of_the_allowed_actions",
      "problem": "single concrete issue",
      "action_reason": "why this action fixes THIS issue",
      "risk_if_ignored": "impact if skipped"
    }}
  ],
  "dataset_level_steps": [
    {{
      "step_id": "remove_duplicates",
      "recommended_action": "REMOVE_DUPLICATES",
      "problem": "dataset-level issue",
      "action_reason": "why this action is best",
      "risk_if_ignored": "impact"
    }}
  ]
}}
"""

# -------------------------------
# Safe JSON extraction
# -------------------------------
def safe_json_parse(text: str):
    if not text:
        return {}

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return {}

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}

# -------------------------------
# API endpoint (ROUTER BASED)
# -------------------------------
@router.post("/llm-review")
def llm_review(request: ReviewRequest):

    user_prompt = USER_PROMPT_TEMPLATE.format(
        metadata=json.dumps(request.dataset_metadata, indent=2),
        rows=json.dumps(request.sample_rows[:20], indent=2)
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content
    parsed = safe_json_parse(raw_output)

    if not parsed:
        return {
            "error": "LLM returned invalid JSON",
            "raw_response": raw_output
        }

    return parsed
