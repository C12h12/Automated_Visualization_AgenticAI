import ast
import os
import re
import time
import uuid
from typing import List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


router = APIRouter(
    prefix="/chart",
    tags=["Chart Generator"]
)


STATIC_DIR = "static"
CLEANED_DATASET_DIR = "cleaned_datasets"
PDF_DIR = "pdfs"

IMAGE_TTL_SECONDS = 300
MAX_CATEGORIES = 12

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(CLEANED_DATASET_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)


sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#0f0a1b",
    "axes.facecolor": "#171124",
    "axes.edgecolor": "#3b3154",
    "axes.labelcolor": "#f4f0ff",
    "xtick.color": "#d6cffa",
    "ytick.color": "#d6cffa",
    "text.color": "#f4f0ff",
    "axes.titlecolor": "#ffffff",
    "grid.color": "#2b2340",
    "grid.alpha": 0.35,
    "savefig.facecolor": "#0f0a1b",
    "savefig.edgecolor": "#0f0a1b",
    "font.size": 10,
})

PRIMARY_COLORS = ["#cf4dff", "#7d5cff", "#59d0ff", "#f46fff", "#6ef3d6", "#ff8ac7"]
SEQUENTIAL_CMAP = sns.color_palette(["#1a1328", "#7d5cff", "#cf4dff", "#ffd0fb"], as_cmap=True)


def clear_static_folder():
    for file in os.listdir(STATIC_DIR):
        path = os.path.join(STATIC_DIR, file)
        if os.path.isfile(path) and file.endswith(".png"):
            os.remove(path)


def cleanup_old_images():
    now = time.time()
    for file in os.listdir(STATIC_DIR):
        path = os.path.join(STATIC_DIR, file)
        if os.path.isfile(path) and file.endswith(".png"):
            if now - os.path.getmtime(path) > IMAGE_TTL_SECONDS:
                os.remove(path)


def get_latest_csv():
    files = [
        os.path.join(CLEANED_DATASET_DIR, f)
        for f in os.listdir(CLEANED_DATASET_DIR)
        if f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError("No cleaned CSV found")
    return max(files, key=os.path.getmtime)


def get_all_current_images():
    return sorted(
        [os.path.join(STATIC_DIR, f) for f in os.listdir(STATIC_DIR) if f.endswith(".png")],
        key=os.path.getmtime
    )


def generate_pdf_from_images(images):
    pdf_name = f"dashboard_{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(PDF_DIR, pdf_name)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    page_width, page_height = A4

    cols, rows = 2, 3
    margin_x, margin_y = 30, 40
    spacing_x, spacing_y = 20, 20

    usable_width = page_width - 2 * margin_x
    usable_height = page_height - 2 * margin_y

    img_width = (usable_width - (cols - 1) * spacing_x) / cols
    img_height = (usable_height - (rows - 1) * spacing_y) / rows

    x_positions = [margin_x + i * (img_width + spacing_x) for i in range(cols)]
    y_positions = [
        page_height - margin_y - img_height - i * (img_height + spacing_y)
        for i in range(rows)
    ]

    for idx, img_path in enumerate(images):
        col = idx % cols
        row = (idx // cols) % rows
        c.drawImage(
            ImageReader(img_path),
            x_positions[col],
            y_positions[row],
            img_width,
            img_height,
            preserveAspectRatio=True
        )
        if (idx + 1) % (cols * rows) == 0:
            c.showPage()

    c.save()
    return pdf_path


def normalize_columns(col_str):
    if isinstance(col_str, list):
        return [str(col).strip().strip("\"'") for col in col_str if str(col).strip()]

    if isinstance(col_str, str):
        col_str = col_str.strip()
        if col_str.startswith("[") and col_str.endswith("]"):
            try:
                parsed = ast.literal_eval(col_str)
                if isinstance(parsed, list):
                    return [str(col).strip().strip("\"'") for col in parsed if str(col).strip()]
            except Exception:
                pass
        if "," in col_str:
            return [part.strip().strip("\"'") for part in col_str.split(",") if part.strip()]
        return [col_str.strip("\"'")]

    return []


def sanitize_filename_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def require_column_count(cols, minimum, chart_name):
    if len(cols) < minimum:
        raise ValueError(
            f"{chart_name} requires at least {minimum} column(s), received {len(cols)}: {cols}"
        )


def resolve_columns(df, cols):
    resolved = []
    lower_map = {str(column).strip().lower(): column for column in df.columns}

    for col in cols:
        cleaned = str(col).strip()
        if cleaned in df.columns:
            resolved.append(cleaned)
            continue

        lowered = cleaned.lower()
        if lowered in lower_map:
            resolved.append(lower_map[lowered])
            continue

        partial_match = next(
            (actual for actual in df.columns if lowered in str(actual).strip().lower()),
            None
        )
        if partial_match is not None:
            resolved.append(partial_match)
            continue

        raise ValueError(
            f"Column '{cleaned}' not found in dataset. Available columns: {list(df.columns)}"
        )

    return resolved


def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def coerce_datetime(series):
    return pd.to_datetime(series, errors="coerce")


def is_numeric_series(series):
    return coerce_numeric(series).notna().sum() > 0


def is_datetime_series(series):
    return coerce_datetime(series).notna().sum() > 0


def first_numeric_column(df, candidates):
    for col in candidates:
        if is_numeric_series(df[col]):
            return col
    return None


def first_datetime_column(df, candidates):
    for col in candidates:
        if is_datetime_series(df[col]):
            return col
    return None


def trim_categories(series, max_categories=MAX_CATEGORIES):
    counts = series.astype(str).fillna("Missing").value_counts()
    if len(counts) <= max_categories:
        return series.astype(str).fillna("Missing")

    keep = set(counts.head(max_categories - 1).index)
    trimmed = series.astype(str).fillna("Missing").apply(lambda value: value if value in keep else "Other")
    return trimmed


def shorten_label(value, max_len=20):
    value = sanitize_filename_text(value)
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


def apply_common_style(ax, title, xlabel=None, ylabel=None, rotate_x=False):
    ax.set_title(sanitize_filename_text(title), fontsize=14, fontweight="bold", pad=14)
    if xlabel is not None:
        ax.set_xlabel(shorten_label(xlabel, 28), labelpad=10)
    if ylabel is not None:
        ax.set_ylabel(shorten_label(ylabel, 28), labelpad=10)

    for spine in ax.spines.values():
        spine.set_color("#3b3154")

    if rotate_x:
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")


def aggregate_series(df, x_col, y_col=None, agg="mean", limit=MAX_CATEGORIES):
    if y_col is None:
        grouped = trim_categories(df[x_col], limit).value_counts().sort_values(ascending=False)
        grouped.index = [shorten_label(value) for value in grouped.index]
        return grouped

    working = df[[x_col, y_col]].copy()
    working[y_col] = coerce_numeric(working[y_col])
    working = working.dropna(subset=[y_col])
    working[x_col] = trim_categories(working[x_col], limit)

    grouped = working.groupby(x_col, dropna=False)[y_col].agg(agg).sort_values(ascending=False)
    grouped.index = [shorten_label(value) for value in grouped.index]
    return grouped.head(limit)


def make_line_data(df, x_col, y_col):
    working = df[[x_col, y_col]].copy()
    working[y_col] = coerce_numeric(working[y_col])
    working = working.dropna(subset=[y_col])

    if is_datetime_series(working[x_col]):
        working[x_col] = coerce_datetime(working[x_col])
        working = working.dropna(subset=[x_col]).sort_values(x_col)
        grouped = working.groupby(x_col)[y_col].mean()
        return grouped.tail(50), True

    working[x_col] = trim_categories(working[x_col], MAX_CATEGORIES)
    grouped = working.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(MAX_CATEGORIES)
    return grouped.sort_index(), False


def infer_chart_type(chart_text):
    chart = sanitize_filename_text(chart_text).lower()

    aliases = [
        ("stacked_bar", ["stacked bar", "stack bar"]),
        ("grouped_bar", ["grouped bar", "clustered bar"]),
        ("bar", ["bar chart", "bar graph", "column chart", "column graph", "bar"]),
        ("line", ["line chart", "line graph", "trend line", "trend chart", "time series", "line"]),
        ("area", ["area chart", "stacked area", "area"]),
        ("scatter", ["scatter plot", "scatter chart", "scatter"]),
        ("heatmap", ["heat map", "heatmap", "correlation map"]),
        ("boxplot", ["box plot", "boxplot", "box and whisker"]),
        ("radar", ["radar chart", "spider chart", "radar"]),
        ("funnel", ["funnel chart", "funnel"]),
        ("pie", ["pie chart", "donut chart", "doughnut chart", "pie", "donut", "doughnut"]),
        ("histogram", ["histogram", "distribution"]),
    ]

    for canonical, patterns in aliases:
        if any(pattern in chart for pattern in patterns):
            return canonical

    raise ValueError(f"Unsupported chart type: {chart_text}")


def plot_bar(df, cols, title):
    require_column_count(cols, 1, "Bar chart")
    cols = resolve_columns(df, cols[:2])
    x = cols[0]
    y = cols[1] if len(cols) > 1 and is_numeric_series(df[cols[1]]) else None

    fig, ax = plt.subplots(figsize=(8, 5.2))
    series = aggregate_series(df, x, y)
    series.plot(kind="bar", color=PRIMARY_COLORS[0], ax=ax)
    ylabel = "Count" if y is None else f"Average {y}"
    apply_common_style(ax, title, x, ylabel, rotate_x=True)
    return fig


def plot_grouped_bar(df, cols, title):
    require_column_count(cols, 2, "Grouped bar chart")
    cols = resolve_columns(df, cols[:3])
    x, group = cols[0], cols[1]
    y = cols[2] if len(cols) > 2 and is_numeric_series(df[cols[2]]) else None

    working = df.copy()
    working[x] = trim_categories(working[x], 8)
    working[group] = trim_categories(working[group], 6)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    if y:
        working[y] = coerce_numeric(working[y])
        working = working.dropna(subset=[y])
        pivot = working.groupby([x, group])[y].mean().unstack(fill_value=0)
    else:
        pivot = pd.crosstab(working[x], working[group])
    pivot.plot(kind="bar", ax=ax, color=PRIMARY_COLORS)
    apply_common_style(ax, title, x, "Value", rotate_x=True)
    ax.legend(title=shorten_label(group, 20), facecolor="#171124", edgecolor="#3b3154")
    return fig


def plot_line(df, cols, title):
    require_column_count(cols, 2, "Line chart")
    cols = resolve_columns(df, cols[:2])
    x, y = cols[0], cols[1]
    if not is_numeric_series(df[y]):
        raise ValueError(f"Line chart requires a numeric y-axis column. Received: {y}")

    series, is_time = make_line_data(df, x, y)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(series.index, series.values, color=PRIMARY_COLORS[2], linewidth=2.5, marker="o", markersize=4)
    ylabel = f"Average {y}"
    apply_common_style(ax, title, x, ylabel, rotate_x=not is_time)
    return fig


def plot_stacked_bar(df, cols, title):
    require_column_count(cols, 2, "Stacked bar chart")
    cols = resolve_columns(df, cols[:3])
    x, group = cols[0], cols[1]
    y = cols[2] if len(cols) > 2 and is_numeric_series(df[cols[2]]) else None

    working = df.copy()
    working[x] = trim_categories(working[x], 8)
    working[group] = trim_categories(working[group], 6)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    if y:
        working[y] = coerce_numeric(working[y])
        working = working.dropna(subset=[y])
        pivot = working.groupby([x, group])[y].sum().unstack(fill_value=0)
    else:
        pivot = pd.crosstab(working[x], working[group])
    pivot.plot(kind="bar", stacked=True, ax=ax, color=PRIMARY_COLORS)
    apply_common_style(ax, title, x, "Value", rotate_x=True)
    ax.legend(title=shorten_label(group, 20), facecolor="#171124", edgecolor="#3b3154")
    return fig


def plot_heatmap(df, cols, title):
    require_column_count(cols, 2, "Heatmap")
    cols = resolve_columns(df, cols[:2])
    x, y = cols[0], cols[1]

    working = df.copy()
    working[x] = trim_categories(working[x], 10)
    working[y] = trim_categories(working[y], 10)
    pivot = pd.crosstab(working[x], working[y])

    fig, ax = plt.subplots(figsize=(8, 5.6))
    sns.heatmap(pivot, annot=True, fmt="g", cmap=SEQUENTIAL_CMAP, linewidths=0.4, cbar=True, ax=ax)
    apply_common_style(ax, title, x, y)
    return fig


def plot_area(df, cols, title):
    require_column_count(cols, 2, "Area chart")
    cols = resolve_columns(df, cols[:3])
    x = cols[0]
    y = first_numeric_column(df, cols[1:]) or cols[-1]
    if not is_numeric_series(df[y]):
        raise ValueError(f"Area chart requires a numeric value column. Received: {y}")

    series, is_time = make_line_data(df, x, y)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.fill_between(range(len(series)), series.values, color=PRIMARY_COLORS[1], alpha=0.55)
    ax.plot(range(len(series)), series.values, color=PRIMARY_COLORS[0], linewidth=2.2)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(series.index)
    apply_common_style(ax, title, x, y, rotate_x=not is_time)
    return fig


def plot_boxplot(df, cols, title):
    require_column_count(cols, 2, "Boxplot")
    cols = resolve_columns(df, cols[:2])
    category_col, value_col = cols[0], cols[1]
    if not is_numeric_series(df[value_col]):
        raise ValueError(f"Boxplot requires numeric values. Received: {value_col}")

    working = df[[category_col, value_col]].copy()
    working[value_col] = coerce_numeric(working[value_col])
    working = working.dropna(subset=[value_col])
    working[category_col] = trim_categories(working[category_col], 8)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    sns.boxplot(data=working, x=category_col, y=value_col, ax=ax, color=PRIMARY_COLORS[1], fliersize=2)
    apply_common_style(ax, title, category_col, value_col, rotate_x=True)
    return fig


def plot_funnel(df, cols, title):
    require_column_count(cols, 2, "Funnel chart")
    cols = resolve_columns(df, cols[:2])
    stage_col, value_col = cols[0], cols[1]
    if not is_numeric_series(df[value_col]):
        raise ValueError(f"Funnel chart requires numeric values. Received: {value_col}")

    series = aggregate_series(df, stage_col, value_col, agg="sum").sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(series.index, series.values, color=PRIMARY_COLORS[0])
    apply_common_style(ax, title, value_col, stage_col)
    return fig


def plot_radar(df, cols, title):
    require_column_count(cols, 3, "Radar chart")
    cols = resolve_columns(df, cols)
    category_col = cols[0]
    metric_cols = [col for col in cols[1:] if is_numeric_series(df[col])]
    if len(metric_cols) < 2:
        raise ValueError("Radar chart requires one category column and at least two numeric metric columns.")

    working = df[[category_col] + metric_cols].copy()
    working[category_col] = trim_categories(working[category_col], 5)
    for metric in metric_cols:
      working[metric] = coerce_numeric(working[metric])
    grouped = working.groupby(category_col)[metric_cols].mean().dropna()

    if grouped.empty:
        raise ValueError("Radar chart could not build a non-empty grouped dataset.")

    grouped = grouped.head(4)
    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6.4, 6.2))
    ax = fig.add_subplot(111, polar=True)
    for index, (_, row) in enumerate(grouped.iterrows()):
        values = row.tolist()
        values += values[:1]
        color = PRIMARY_COLORS[index % len(PRIMARY_COLORS)]
        ax.plot(angles, values, linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([shorten_label(metric, 16) for metric in metric_cols])
    ax.set_title(sanitize_filename_text(title), fontsize=14, fontweight="bold", pad=18)
    ax.set_facecolor("#171124")
    return fig


def plot_scatter(df, cols, title):
    require_column_count(cols, 2, "Scatter chart")
    cols = resolve_columns(df, cols[:3])
    x, y = cols[0], cols[1]
    if not is_numeric_series(df[x]) or not is_numeric_series(df[y]):
        raise ValueError(f"Scatter chart requires two numeric columns. Received: {cols[:2]}")

    working = df[[x, y]].copy()
    working[x] = coerce_numeric(working[x])
    working[y] = coerce_numeric(working[y])
    working = working.dropna().head(1000)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(working[x], working[y], alpha=0.72, color=PRIMARY_COLORS[2], edgecolors="none")
    apply_common_style(ax, title, x, y)
    return fig


def plot_pie(df, cols, title):
    require_column_count(cols, 1, "Pie chart")
    cols = resolve_columns(df, cols[:2])
    category_col = cols[0]
    value_col = cols[1] if len(cols) > 1 and is_numeric_series(df[cols[1]]) else None

    if value_col:
        series = aggregate_series(df, category_col, value_col, agg="sum", limit=6)
    else:
        series = aggregate_series(df, category_col, None, limit=6)

    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.pie(
        series.values,
        labels=series.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=PRIMARY_COLORS[: len(series)],
        textprops={"color": "#ffffff", "fontsize": 9}
    )
    ax.set_title(sanitize_filename_text(title), fontsize=14, fontweight="bold", pad=14)
    return fig


def plot_histogram(df, cols, title):
    require_column_count(cols, 1, "Histogram")
    cols = resolve_columns(df, cols[:1])
    value_col = cols[0]
    if not is_numeric_series(df[value_col]):
        raise ValueError(f"Histogram requires a numeric column. Received: {value_col}")

    values = coerce_numeric(df[value_col]).dropna()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.hist(values, bins=min(20, max(8, int(np.sqrt(len(values))))) if len(values) else 10, color=PRIMARY_COLORS[0], edgecolor="#f4f0ff")
    apply_common_style(ax, title, value_col, "Frequency")
    return fig


def build_fallback_chart_candidates(df, cols):
    resolved = []
    try:
        resolved = resolve_columns(df, cols)
    except Exception:
        resolved = []

    numeric_cols = [col for col in resolved if is_numeric_series(df[col])]
    categorical_cols = [col for col in resolved if not is_numeric_series(df[col])]
    datetime_cols = [col for col in resolved if is_datetime_series(df[col])]

    all_numeric = [col for col in df.columns if is_numeric_series(df[col])]
    all_datetime = [col for col in df.columns if is_datetime_series(df[col])]
    all_categorical = [col for col in df.columns if col not in all_numeric]

    candidates = []

    if resolved:
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            candidates.extend([
                ("bar", [categorical_cols[0], numeric_cols[0]]),
                ("stacked_bar", resolved[: min(3, len(resolved))]),
                ("boxplot", [categorical_cols[0], numeric_cols[0]]),
                ("funnel", [categorical_cols[0], numeric_cols[0]]),
                ("pie", [categorical_cols[0], numeric_cols[0]]),
            ])
        if len(numeric_cols) >= 2:
            candidates.extend([
                ("scatter", numeric_cols[:2]),
                ("histogram", [numeric_cols[0]]),
            ])
        if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
            candidates.extend([
                ("line", [datetime_cols[0], numeric_cols[0]]),
                ("area", [datetime_cols[0], numeric_cols[0]]),
            ])
        if len(categorical_cols) >= 2:
            candidates.extend([
                ("heatmap", categorical_cols[:2]),
                ("stacked_bar", categorical_cols[:2]),
            ])
        if len(categorical_cols) >= 1:
            candidates.extend([
                ("bar", [categorical_cols[0]]),
                ("pie", [categorical_cols[0]]),
            ])

    if not candidates:
        if all_datetime and all_numeric:
            candidates.append(("line", [all_datetime[0], all_numeric[0]]))
        if all_categorical and all_numeric:
            candidates.append(("bar", [all_categorical[0], all_numeric[0]]))
        if len(all_numeric) >= 2:
            candidates.append(("scatter", all_numeric[:2]))
        if all_numeric:
            candidates.append(("histogram", [all_numeric[0]]))
        if len(all_categorical) >= 2:
            candidates.append(("heatmap", all_categorical[:2]))
        if all_categorical:
            candidates.append(("bar", [all_categorical[0]]))

    deduped = []
    seen = set()
    for chart_type, candidate_cols in candidates:
        key = (chart_type, tuple(candidate_cols))
        if key not in seen and candidate_cols:
            seen.add(key)
            deduped.append((chart_type, candidate_cols))

    return deduped


def build_error_figure(title, message):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.set_facecolor("#171124")
    fig.patch.set_facecolor("#0f0a1b")
    ax.axis("off")

    ax.text(
        0.5, 0.72, sanitize_filename_text(title),
        ha="center", va="center",
        fontsize=15, fontweight="bold", color="#ffffff",
        wrap=True, transform=ax.transAxes
    )
    ax.text(
        0.5, 0.46, "Chart fallback could not fully match this KPI.",
        ha="center", va="center",
        fontsize=11, color="#d6cffa",
        transform=ax.transAxes
    )
    ax.text(
        0.5, 0.24, sanitize_filename_text(message)[:180],
        ha="center", va="center",
        fontsize=9.5, color="#ffbfd0",
        wrap=True, transform=ax.transAxes
    )
    return fig


def generate_chart_by_type(df, chart_type, cols, title):
    if chart_type == "stacked_bar":
        return plot_stacked_bar(df, cols, title)
    if chart_type == "grouped_bar":
        return plot_grouped_bar(df, cols, title)
    if chart_type == "bar":
        return plot_bar(df, cols, title)
    if chart_type == "line":
        return plot_line(df, cols, title)
    if chart_type == "area":
        return plot_area(df, cols, title)
    if chart_type == "scatter":
        return plot_scatter(df, cols, title)
    if chart_type == "heatmap":
        return plot_heatmap(df, cols, title)
    if chart_type == "boxplot":
        return plot_boxplot(df, cols, title)
    if chart_type == "radar":
        return plot_radar(df, cols, title)
    if chart_type == "funnel":
        return plot_funnel(df, cols, title)
    if chart_type == "pie":
        return plot_pie(df, cols, title)
    if chart_type == "histogram":
        return plot_histogram(df, cols, title)
    raise ValueError(f"Unsupported chart type: {chart_type}")


def save_figure(fig, filepath):
    fig.tight_layout()
    fig.savefig(filepath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_chart(df, kpi, filepath):
    cols = normalize_columns(kpi.columns)
    errors = []
    attempted = []

    try:
        suggested_type = infer_chart_type(kpi.suggested_chart)
        attempted.append((suggested_type, cols))
    except Exception as e:
        errors.append(str(e))

    attempted.extend(build_fallback_chart_candidates(df, cols))

    seen = set()
    for chart_type, candidate_cols in attempted:
        key = (chart_type, tuple(candidate_cols))
        if key in seen:
            continue
        seen.add(key)
        try:
            fig = generate_chart_by_type(df, chart_type, candidate_cols, kpi.name)
            save_figure(fig, filepath)
            return None
        except Exception as e:
            errors.append(f"{chart_type} with columns {candidate_cols}: {e}")
            plt.close("all")

    fallback_fig = build_error_figure(kpi.name, errors[-1] if errors else "Unknown chart generation error")
    save_figure(fallback_fig, filepath)
    return f"Fallback chart used for KPI '{kpi.name}'. Last error: {errors[-1] if errors else 'Unknown error'}"


class KPI(BaseModel):
    name: str
    columns: str
    suggested_chart: str


class KPIRequest(BaseModel):
    kpis: List[KPI]


@router.post("/generate")
def generate_charts(req: KPIRequest):
    clear_static_folder()
    cleanup_old_images()

    df = pd.read_csv(get_latest_csv())
    images = []
    errors = []

    for kpi in req.kpis:
        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join(STATIC_DIR, filename)
        try:
            warning = plot_chart(df, kpi, path)
            images.append(f"/static/{filename}")
            if warning:
                print(warning)
                errors.append(warning)
        except Exception as e:
            error_message = f"Error generating '{kpi.name}': {e}"
            print(error_message)
            errors.append(error_message)

    if not images:
        return JSONResponse(
            status_code=422,
            content={"images": [], "errors": errors or ["No charts could be generated."]}
        )

    return {"images": images, "errors": errors}


@router.get("/download-dashboard")
def download_dashboard():
    images = get_all_current_images()
    if not images:
        return {"error": "No charts available"}

    pdf_path = generate_pdf_from_images(images)
    return FileResponse(pdf_path, media_type="application/pdf", filename="Dashboard_Report.pdf")
