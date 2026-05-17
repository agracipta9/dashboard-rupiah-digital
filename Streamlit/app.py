from __future__ import annotations

import html
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from config import (
    APP_SUBTITLE, APP_TITLE, LABEL_COLORS, LABEL_ORDER, MODEL_CONFIG,
    RESULT_FILES, RM2_CATEGORY_ORDER, THEME,
)
from utils import (
    clean_unnamed_columns, find_existing_path, format_metric_value, format_percent,
    get_prediction_interpretation, get_preprocessing_description, load_first_existing,
    load_indobert_model, load_sklearn_pipeline, load_table_safely, normalize_label,
    normalize_label_series, predict_with_indobert, predict_with_sklearn_pipeline,
    read_excel_sheets_safely, show_image_or_warning,
)

st.set_page_config(
    page_title="Dashboard Analisis Sentimen Rupiah Digital",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = f"""
<style>
:root {{
    --page-bg: {THEME["page_bg"]};
    --card-bg: {THEME["card_bg"]};
    --sidebar-bg: {THEME["sidebar_bg"]};
    --primary: {THEME["primary"]};
    --secondary: {THEME["secondary"]};
    --text: {THEME["text"]};
    --muted: {THEME["muted"]};
    --border: {THEME["border"]};
    --plot-bg: {THEME["plot_bg"]};
}}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {{
    display: none !important;
}}

#MainMenu, footer {{
    visibility: hidden !important;
}}

.stApp {{
    background: var(--page-bg);
    color: var(--text);
}}

.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: min(1680px, 96vw) !important;
    padding-left: 1.6rem !important;
    padding-right: 1.6rem !important;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
}}

[data-testid="stSidebar"] .block-container {{
    padding-top: 1.25rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}}

[data-testid="stSidebar"] * {{
    color: #E5E7EB;
}}

.sidebar-title {{
    font-size: 1.05rem;
    font-weight: 850;
    line-height: 1.25;
    margin-bottom: 0.25rem;
    color: #FFFFFF;
}}

.sidebar-subtitle {{
    font-size: 0.76rem;
    line-height: 1.45;
    color: #CBD5E1;
    margin-bottom: 0.75rem;
}}

.sidebar-summary {{
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: rgba(30, 41, 59, 0.68);
    border-radius: 14px;
    padding: 0.72rem 0.78rem;
    margin: 0.7rem 0 0.9rem 0;
}}

.sidebar-summary-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.55rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.16);
}}

.sidebar-summary-row:last-child {{
    border-bottom: none;
}}

.sidebar-summary-row span {{
    color: #CBD5E1;
    font-size: 0.74rem;
}}

.sidebar-summary-row strong {{
    color: #FFFFFF;
    font-size: 0.87rem;
    font-weight: 800;
}}

[data-testid="stSidebar"] [role="radiogroup"] {{
    gap: 0.18rem;
}}

[data-testid="stSidebar"] label {{
    background: transparent !important;
    border-radius: 10px;
    padding: 0.12rem 0.2rem;
}}

[data-testid="stSidebar"] label:hover {{
    background: rgba(255, 255, 255, 0.07) !important;
}}

[data-testid="stSidebar"] label p {{
    font-size: 0.92rem !important;
    font-weight: 650 !important;
    color: #E5E7EB !important;
}}

.page-header, .section-box, .metric-card, .comment-card, .prediction-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    color: var(--text);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.25);
}}

.page-header {{
    border-radius: 20px;
    padding: 1.25rem 1.45rem;
    margin-bottom: 1rem;
}}

.page-title {{
    font-size: clamp(1.45rem, 2.3vw, 2.05rem);
    font-weight: 850;
    color: #F8FAFC;
    margin: 0 0 0.35rem 0;
}}

.page-subtitle {{
    color: #CBD5E1;
    font-size: 0.98rem;
    line-height: 1.6;
    margin: 0;
}}

.section-box {{
    border-radius: 18px;
    padding: 1rem 1.12rem;
    margin: 0.9rem 0 1.05rem 0;
}}

.section-title, h1, h2, h3 {{
    color: #F8FAFC !important;
}}

.section-subtitle, .metric-label, .metric-note, .small-caption {{
    color: #CBD5E1 !important;
}}

.metric-card {{
    border-radius: 18px;
    padding: 1rem 1rem;
    min-height: 132px;
    height: 132px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}

.metric-label {{
    font-size: 0.82rem;
    margin-bottom: 0.45rem;
    font-weight: 700;
    line-height: 1.3;
}}

.metric-value {{
    color: #F8FAFC;
    font-size: 1.42rem;
    font-weight: 850;
    line-height: 1.22;
}}

.summary-box, .info-box {{
    background: #172554;
    border: 1px solid #2563EB;
    color: #E0F2FE;
    border-radius: 15px;
    padding: 0.95rem 1.05rem;
    line-height: 1.65;
    margin: 0.75rem 0;
}}

.comment-card {{
    border-radius: 15px;
    padding: 0.85rem 0.95rem;
    margin: 0.5rem 0;
}}

.comment-label {{
    display: inline-block;
    font-size: 0.76rem;
    font-weight: 800;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    margin-bottom: 0.5rem;
}}

.label-negatif {{ background: #7F1D1D; color: #FCA5A5; }}
.label-netral {{ background: #334155; color: #E2E8F0; }}
.label-positif {{ background: #064E3B; color: #86EFAC; }}

.comment-text {{
    color: #E5E7EB;
    line-height: 1.62;
    font-size: 0.94rem;
}}

.prediction-card {{
    border-radius: 18px;
    padding: 1.15rem 1.25rem;
    margin-top: 1rem;
}}

.prediction-label {{
    font-size: 1.72rem;
    font-weight: 850;
    margin-bottom: 0.35rem;
}}

.pred-negatif {{ color: #F87171; }}
.pred-netral {{ color: #CBD5E1; }}
.pred-positif {{ color: #4ADE80; }}

.divider {{
    height: 1px;
    background: var(--border);
    margin: 1.2rem 0;
}}

.light-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: #111827;
    border: 1px solid #334155;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 7px 18px rgba(0, 0, 0, 0.25);
}}

.light-table th {{
    background: #1E293B;
    color: #CBD5E1;
    font-weight: 800;
    font-size: 0.84rem;
    padding: 0.72rem 0.8rem;
    border-bottom: 1px solid #334155;
    text-align: left;
}}

.light-table td {{
    color: #F8FAFC;
    font-size: 0.88rem;
    padding: 0.66rem 0.8rem;
    border-bottom: 1px solid #263244;
}}

.light-table tr:last-child td {{
    border-bottom: none;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid #334155;
    border-radius: 14px;
    overflow: hidden;
}}

.stTextArea textarea,
.stSelectbox [data-baseweb="select"] {{
    background-color: #111827 !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
}}

.stTextArea label,
.stSelectbox label {{
    color: #CBD5E1 !important;
    font-weight: 700 !important;
}}

button[kind="primary"] {{
    background: #2563EB !important;
    border-color: #2563EB !important;
}}

.metric-mini-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
}}

@media (max-width: 900px) {{
    .metric-mini-grid {{
        grid-template-columns: 1fr;
    }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="page-header">
    <div class="page-title">{html.escape(title)}</div>
    <p class="page-subtitle">{html.escape(subtitle)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str | None = None) -> None:
    note_html = f'<div class="metric-note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{html.escape(label)}</div>
    <div class="metric-value">{html.escape(value)}</div>
    {note_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_summary_box(text: str) -> None:
    st.markdown(f'<div class="summary-box">{text}</div>', unsafe_allow_html=True)


def render_info_box(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def render_comment_card(sentiment: str, comment: str) -> None:
    sentiment = normalize_label(sentiment)
    css_class = {
        "Negatif": "label-negatif",
        "Netral": "label-netral",
        "Positif": "label-positif",
    }.get(sentiment, "label-netral")
    st.markdown(
        f"""
<div class="comment-card">
    <span class="comment-label {css_class}">{html.escape(sentiment)}</span>
    <div class="comment-text">{html.escape(str(comment))}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def dataframe_to_dark_html(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "<p>Data belum tersedia.</p>"
    temp = df.head(max_rows).copy()
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in temp.columns)
    rows = []
    for _, row in temp.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row.values)
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="light-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def display_dark_table(df: pd.DataFrame, max_rows: int = 10) -> None:
    st.markdown(dataframe_to_dark_html(df, max_rows=max_rows), unsafe_allow_html=True)


def display_dataframe(df: pd.DataFrame, hide_index: bool = True):
    if df.empty:
        st.info("Data belum tersedia.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)


@st.cache_data(show_spinner=False)
def load_labeled_data() -> pd.DataFrame:
    df = load_table_safely(RESULT_FILES["labeled_data"], "Dataset final berlabel")
    if not df.empty and "final_label" in df.columns:
        df["final_label"] = normalize_label_series(df["final_label"])
    return df


@st.cache_data(show_spinner=False)
def load_analysis_table(key: str) -> pd.DataFrame:
    return load_table_safely(RESULT_FILES[key], key)


def make_sentiment_distribution(labeled_df: pd.DataFrame) -> pd.DataFrame:
    dist = load_analysis_table("final_sentiment_distribution")
    if not dist.empty:
        dist = dist.copy()
        if "final_label" in dist.columns:
            dist = dist.rename(columns={"final_label": "label"})
        if "label" not in dist.columns and len(dist.columns) > 0:
            dist = dist.rename(columns={dist.columns[0]: "label"})
        if "jumlah" not in dist.columns:
            numeric_cols = [c for c in dist.columns if pd.api.types.is_numeric_dtype(dist[c])]
            if numeric_cols:
                dist = dist.rename(columns={numeric_cols[0]: "jumlah"})
        dist["label"] = normalize_label_series(dist["label"])
    elif not labeled_df.empty and "final_label" in labeled_df.columns:
        dist = labeled_df["final_label"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int).reset_index()
        dist.columns = ["label", "jumlah"]
    else:
        dist = pd.DataFrame({"label": LABEL_ORDER, "jumlah": [0, 0, 0]})

    dist = dist[["label", "jumlah"]].groupby("label", as_index=False)["jumlah"].sum()
    dist["label"] = pd.Categorical(dist["label"], categories=LABEL_ORDER, ordered=True)
    dist = dist.sort_values("label").reset_index(drop=True)
    total = dist["jumlah"].sum()
    dist["persentase"] = np.where(total > 0, dist["jumlah"] / total * 100, 0).round(2)
    return dist


def make_model_comparison() -> pd.DataFrame:
    df = load_analysis_table("model_comparison_summary")
    if df.empty:
        df, _ = load_first_existing(
            [RESULT_FILES["evaluation_results"], RESULT_FILES["evaluation_results_alt"]],
            "evaluation_results",
        )
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        if cl in ["model", "nama_model"]:
            rename_map[c] = "model"
        elif cl in ["accuracy", "akurasi"]:
            rename_map[c] = "accuracy"
        elif cl in ["precision_macro", "macro_precision", "precision_avg_macro"]:
            rename_map[c] = "precision_macro"
        elif cl in ["recall_macro", "macro_recall", "recall_avg_macro"]:
            rename_map[c] = "recall_macro"
        elif cl in ["f1_macro", "macro_f1", "f1_score_macro", "f1_macro_score"]:
            rename_map[c] = "f1_macro"
        elif cl in ["precision_weighted", "precision__weighted"]:
            rename_map[c] = "precision_weighted"
        elif cl in ["recall_weighted", "recall__weighted"]:
            rename_map[c] = "recall_weighted"
        elif cl in ["f1_weighted", "weighted_f1", "f1_weighted_score"]:
            rename_map[c] = "f1_weighted"

    df = df.rename(columns=rename_map)
    if "model" not in df.columns:
        df = df.rename(columns={df.columns[0]: "model"})

    for col in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "f1_macro" in df.columns:
        df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return df


def make_best_model_summary(evaluation_df: pd.DataFrame) -> dict:
    if not evaluation_df.empty:
        metric = "f1_macro" if "f1_macro" in evaluation_df.columns else "accuracy"
        row = evaluation_df.sort_values(metric, ascending=False).iloc[0]
        return {
            "model": str(row.get("model", "IndoBERT")),
            "metric": metric,
            "value": float(row.get(metric, 0)),
            "accuracy": float(row.get("accuracy", 0)) if "accuracy" in evaluation_df.columns else 0.0,
        }
    return {"model": "IndoBERT", "metric": "f1_macro", "value": 0.0, "accuracy": 0.0}


def load_rm2_distribution(identified_only: bool = False) -> pd.DataFrame:
    key = "cash_response_distribution_without_unidentified" if identified_only else "cash_response_distribution"
    df = load_analysis_table(key)

    if df.empty and not identified_only:
        labeled_df = load_labeled_data()
        if labeled_df.empty or "cash_response_category_refined" not in labeled_df.columns:
            return pd.DataFrame()
        exploded = labeled_df.copy()
        exploded["cash_response_category_refined"] = exploded["cash_response_category_refined"].fillna("Tidak teridentifikasi").astype(str)
        exploded = exploded.assign(kategori=exploded["cash_response_category_refined"].str.split("; ")).explode("kategori")
        df = exploded["kategori"].value_counts().reset_index()
        df.columns = ["kategori", "jumlah"]
        df["persentase_terhadap_total_komentar"] = (df["jumlah"] / len(labeled_df) * 100).round(2)

    if not df.empty:
        df = df.copy()
        if "rm2_category" in df.columns and "kategori" not in df.columns:
            df = df.rename(columns={"rm2_category": "kategori"})
        if "jumlah" not in df.columns:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                df = df.rename(columns={numeric_cols[0]: "jumlah"})
        if identified_only:
            df = df[df["kategori"] != "Tidak teridentifikasi"].copy()
        df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0).astype(int)
        df = df.sort_values("jumlah", ascending=False).reset_index(drop=True)
    return df


def apply_dark_plot_layout(fig, height=440):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=20),
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font=dict(color="#F8FAFC"),
        xaxis=dict(tickfont=dict(color="#F8FAFC"), title_font=dict(color="#F8FAFC"), gridcolor="#334155"),
        yaxis=dict(tickfont=dict(color="#F8FAFC"), title_font=dict(color="#F8FAFC"), gridcolor="#334155"),
        legend=dict(font=dict(color="#F8FAFC")),
    )
    return fig


def plot_sentiment_bar(dist_df: pd.DataFrame, title: str):
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            dist_df, x="label", y="jumlah", text="jumlah", color="label",
            color_discrete_map=LABEL_COLORS,
            category_orders={"label": LABEL_ORDER},
            title=title,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Label Sentimen", yaxis_title="Jumlah Komentar")
        st.plotly_chart(apply_dark_plot_layout(fig, 410), use_container_width=True)
    else:
        st.bar_chart(dist_df.set_index("label")["jumlah"])


def plot_horizontal_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str = "#2563EB", height: int = 480):
    if df.empty:
        st.info("Data belum tersedia.")
        return

    if PLOTLY_AVAILABLE:
        fig = px.bar(df, x=x, y=y, orientation="h", text=x, title=title)
        fig.update_traces(marker_color=color, textposition="outside")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), xaxis_title="Jumlah", yaxis_title="")
        st.plotly_chart(apply_dark_plot_layout(fig, height), use_container_width=True)
    else:
        st.bar_chart(df.set_index(y)[x])


def plot_model_metrics(df: pd.DataFrame):
    if df.empty:
        st.info("Data evaluasi model belum tersedia.")
        return

    metric_cols = [c for c in ["accuracy", "precision_macro", "recall_macro", "f1_macro"] if c in df.columns]
    if not metric_cols:
        st.info("Kolom metrik belum tersedia.")
        return

    long_df = df[["model"] + metric_cols].melt(id_vars="model", value_vars=metric_cols, var_name="Metrik", value_name="Nilai")
    metric_name = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision Macro",
        "recall_macro": "Recall Macro",
        "f1_macro": "F1 Macro",
    }
    long_df["Metrik"] = long_df["Metrik"].map(metric_name).fillna(long_df["Metrik"])

    if PLOTLY_AVAILABLE:
        fig = px.bar(
            long_df, x="Metrik", y="Nilai", color="model", barmode="group",
            text=long_df["Nilai"].round(4),
            title="Perbandingan Performa 3 Model",
            color_discrete_sequence=["#38BDF8", "#F59E0B", "#22C55E"],
        )
        fig.update_traces(textposition="outside", texttemplate="%{text:.4f}")
        fig.update_layout(yaxis_range=[0, 1.05], xaxis_title="Metrik", yaxis_title="Nilai")
        st.plotly_chart(apply_dark_plot_layout(fig, 470), use_container_width=True)
    else:
        st.bar_chart(df.set_index("model")[metric_cols])


def plot_model_heatmap(df: pd.DataFrame):
    if df.empty or not PLOTLY_AVAILABLE:
        return

    metric_cols = [c for c in ["accuracy", "precision_macro", "recall_macro", "f1_macro"] if c in df.columns]
    if not metric_cols:
        return

    heat_df = df[["model"] + metric_cols].copy()
    heat_df = heat_df.set_index("model")
    heat_df.columns = ["Accuracy", "Precision Macro", "Recall Macro", "F1 Macro"][:len(heat_df.columns)]

    fig = px.imshow(
        heat_df,
        text_auto=".4f",
        color_continuous_scale="Blues",
        title="Heatmap Performa Model",
        aspect="auto",
    )
    fig.update_layout(coloraxis_showscale=True)
    st.plotly_chart(apply_dark_plot_layout(fig, 360), use_container_width=True)


def render_model_cards(eval_df: pd.DataFrame, best: dict):
    if eval_df.empty:
        return
    best_row = eval_df.sort_values("f1_macro" if "f1_macro" in eval_df.columns else "accuracy", ascending=False).iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Model Terbaik", str(best_row.get("model", best["model"])))
    with c2:
        render_metric_card("Accuracy Terbaik", format_metric_value(best_row.get("accuracy", 0), 4))
    with c3:
        render_metric_card("F1 Macro Terbaik", format_metric_value(best_row.get("f1_macro", best["value"]), 4))


def model_summary_for_overview(eval_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"] if c in eval_df.columns]
    if not cols:
        return pd.DataFrame()
    df = eval_df[cols].copy()
    rename_cols = {
        "model": "Model",
        "accuracy": "Accuracy",
        "precision_macro": "Precision Macro",
        "recall_macro": "Recall Macro",
        "f1_macro": "F1 Macro",
    }
    df = df.rename(columns=rename_cols)
    for col in ["Accuracy", "Precision Macro", "Recall Macro", "F1 Macro"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{float(x):.4f}")
    return df


def render_sentiment_examples(labeled_df: pd.DataFrame):
    sheets = read_excel_sheets_safely(RESULT_FILES["examples_by_sentiment"])
    if sheets:
        for label in LABEL_ORDER:
            sheet_df = sheets.get(label)
            if sheet_df is None:
                continue
            with st.expander(f"Contoh Komentar {label}", expanded=False):
                for _, row in sheet_df.head(5).iterrows():
                    comment = row.get("Komentar", row.get("display_comment", row.get("clean_comment", "")))
                    sentiment = row.get("Sentimen", label)
                    render_comment_card(sentiment, comment)
        return

    if labeled_df.empty:
        st.info("Contoh komentar belum tersedia.")
        return

    comment_col = next((c for c in ["Comment", "clean_comment", "label_text"] if c in labeled_df.columns), None)
    if comment_col is None:
        st.info("Kolom komentar tidak ditemukan.")
        return

    for label in LABEL_ORDER:
        with st.expander(f"Contoh Komentar {label}", expanded=False):
            temp = labeled_df[labeled_df["final_label"] == label].drop_duplicates(subset=[comment_col]).head(5)
            for _, row in temp.iterrows():
                render_comment_card(label, row[comment_col])


def render_rm2_examples():
    sheets = read_excel_sheets_safely(RESULT_FILES["examples_by_rm2"])
    if not sheets:
        st.info("File contoh komentar per kategori RM2 belum tersedia. Jalankan Analysis_Result.ipynb terlebih dahulu.")
        return

    sheet_map = {name.lower(): (name, df) for name, df in sheets.items()}

    for category in RM2_CATEGORY_ORDER:
        if category == "Tidak teridentifikasi":
            continue

        match = None
        for sheet_name, pair in sheet_map.items():
            if category.lower()[:25] in sheet_name or sheet_name in category.lower():
                match = pair
                break

        if match is None:
            continue

        _, df = match
        with st.expander(category, expanded=False):
            if df.empty:
                st.info("Belum ada contoh komentar untuk kategori ini.")
            else:
                for _, row in df.head(5).iterrows():
                    sentiment = row.get("Sentimen", row.get("final_label", "Netral"))
                    comment = row.get("Komentar", row.get("display_comment", row.get("clean_comment", "")))
                    render_comment_card(sentiment, comment)


def build_sidebar(best: dict, total_comments: int) -> str:
    st.sidebar.markdown(
        """
<div class="sidebar-title">Dashboard Sentimen</div>
<div class="sidebar-subtitle">Rupiah Digital & Uang Tunai</div>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
<div class="sidebar-summary">
    <div class="sidebar-summary-row">
        <span>Total Data</span>
        <strong>{format_metric_value(total_comments, 0)}</strong>
    </div>
    <div class="sidebar-summary-row">
        <span>Model</span>
        <strong>{html.escape(str(best["model"]))}</strong>
    </div>
    <div class="sidebar-summary-row">
        <span>F1 Macro</span>
        <strong>{format_metric_value(best["value"], 3)}</strong>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "Menu",
        ["Overview", "Distribusi Sentimen", "Bentuk Kekhawatiran & Tanggapan", "Perbandingan Model", "Simulasi"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Dashboard membaca file hasil labelling, analysis result, dan modelling.")
    return menu


def tab_overview(labeled_df, sentiment_dist, eval_df, rm2_identified, best):
    render_page_header(APP_TITLE, APP_SUBTITLE)

    render_summary_box(
        "Dataset terdiri dari komentar YouTube yang telah melalui proses <em>cleaning</em>, deduplikasi, "
        "<em>labelling</em>, dan validasi. Sentimen diklasifikasikan ke dalam Negatif, Netral, dan Positif. "
        "Model Naive Bayes, SVM, dan IndoBERT dibandingkan untuk menentukan model terbaik."
    )

    total_comments = int(len(labeled_df)) if not labeled_df.empty else int(sentiment_dist["jumlah"].sum())
    dominant_sentiment = sentiment_dist.sort_values("jumlah", ascending=False).iloc[0]
    negative_count = int(sentiment_dist.loc[sentiment_dist["label"] == "Negatif", "jumlah"].sum())
    dominant_rm2 = rm2_identified.iloc[0]["kategori"] if not rm2_identified.empty else "-"

    row1 = st.columns(3)
    with row1[0]:
        render_metric_card("Total Komentar Dianalisis", format_metric_value(total_comments, 0), "Setelah cleaning, deduplikasi, dan labelling")
    with row1[1]:
        render_metric_card("Sentimen Dominan", str(dominant_sentiment["label"]))
    with row1[2]:
        render_metric_card("Komentar Negatif", format_metric_value(negative_count, 0))

    st.markdown('<div style="height: 0.85rem;"></div>', unsafe_allow_html=True)

    row2 = st.columns(3)
    with row2[0]:
        render_metric_card("Respons Uang Tunai Dominan", str(dominant_rm2))
    with row2[1]:
        render_metric_card("Model Terbaik", str(best["model"]))
    with row2[2]:
        render_metric_card("F1 Macro", format_metric_value(best["value"], 3))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.05, 1.15])
    with col_left:
        st.subheader("Distribusi Sentimen Final")
        plot_sentiment_bar(sentiment_dist, "Distribusi Sentimen Final")

    with col_right:
        st.subheader("Tabel Jumlah Label")
        label_table = sentiment_dist.copy()
        label_table["persentase"] = label_table["persentase"].apply(format_percent)
        display_dark_table(label_table, max_rows=10)

        st.subheader("Ringkasan Performa Model")
        model_table = model_summary_for_overview(eval_df)
        display_dark_table(model_table, max_rows=5)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Temuan Utama")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_info_box(f"<strong>Distribusi Sentimen.</strong><br>Sentimen didominasi <strong>{dominant_sentiment['label']}</strong>.")
    with c2:
        render_info_box(f"<strong>Bentuk Kekhawatiran & Tanggapan.</strong><br>Kategori dominan: <strong>{html.escape(str(dominant_rm2))}</strong>.")
    with c3:
        render_info_box(f"<strong>Perbandingan Model.</strong><br><strong>{html.escape(str(best['model']))}</strong> menjadi model terbaik.")

    with st.expander("Detail Data dan Model", expanded=False):
        train_size = round(total_comments * 0.8)
        test_size = total_comments - train_size
        detail_df = pd.DataFrame([
            {"Komponen": "Komentar dianalisis", "Nilai": format_metric_value(total_comments, 0)},
            {"Komponen": "Data latih", "Nilai": format_metric_value(train_size, 0)},
            {"Komponen": "Data uji", "Nilai": format_metric_value(test_size, 0)},
            {"Komponen": "Jumlah label", "Nilai": "3"},
            {"Komponen": "Jumlah model", "Nilai": "3"},
            {"Komponen": "Model terbaik", "Nilai": str(best["model"])},
        ])
        display_dark_table(detail_df, max_rows=10)


def tab_rm1(labeled_df, sentiment_dist):
    render_page_header("Distribusi Sentimen Pengguna YouTube", "Menampilkan distribusi sentimen akhir pengguna YouTube terhadap Rupiah Digital dan keberlanjutan uang tunai.")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        plot_sentiment_bar(sentiment_dist, "Distribusi Sentimen Akhir")
    with c2:
        display_df = sentiment_dist.copy()
        display_df["persentase"] = display_df["persentase"].apply(format_percent)
        display_dark_table(display_df)
        dominant = sentiment_dist.sort_values("jumlah", ascending=False).iloc[0]
        render_summary_box(f"Sentimen paling dominan adalah <strong>{dominant['label']}</strong>.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Contoh Komentar per Sentimen")
    render_sentiment_examples(labeled_df)


def tab_rm2(rm2_all, rm2_identified):
    render_page_header("Bentuk Kekhawatiran & Tanggapan", "Grafik utama menampilkan kategori yang teridentifikasi agar bentuk kekhawatiran dan tanggapan pengguna YouTube terhadap keberlanjutan penggunaan uang tunai terlihat lebih jelas.")
    total_identified = int(rm2_identified["jumlah"].sum()) if not rm2_identified.empty else 0
    unidentified_count = 0
    if not rm2_all.empty and "Tidak teridentifikasi" in rm2_all["kategori"].values:
        unidentified_count = int(rm2_all.loc[rm2_all["kategori"] == "Tidak teridentifikasi", "jumlah"].iloc[0])
    dominant_category = rm2_identified.iloc[0]["kategori"] if not rm2_identified.empty else "-"

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Kategori Teridentifikasi", format_metric_value(total_identified, 0), "Total kemunculan kategori")
    with c2:
        render_metric_card("Tidak Teridentifikasi", format_metric_value(unidentified_count, 0), "Tidak ditampilkan di grafik utama")
    with c3:
        render_metric_card("Respons Dominan", str(dominant_category))

    render_summary_box("Kategori <strong>Tidak teridentifikasi</strong> tidak ditampilkan pada grafik utama karena tidak memuat indikator eksplisit terkait respons pengguna terhadap keberlanjutan uang tunai.")

    st.subheader("Distribusi Bentuk Kekhawatiran & Tanggapan")
    if rm2_identified.empty:
        st.info("Data kategori teridentifikasi belum tersedia.")
    else:
        plot_horizontal_bar(rm2_identified.sort_values("jumlah", ascending=True), "jumlah", "kategori", "Distribusi Kategori Teridentifikasi", "#2563EB", 520)
        table = rm2_identified.copy()
        for col in table.columns:
            if "persentase" in col:
                table[col] = table[col].apply(format_percent)
        display_dark_table(table, max_rows=15)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Hubungan Sentimen dengan Bentuk Kekhawatiran & Tanggapan")
    sentiment_by_rm2 = load_analysis_table("sentiment_by_cash_category")
    if not sentiment_by_rm2.empty:
        if "rm2_category" in sentiment_by_rm2.columns:
            sentiment_by_rm2 = sentiment_by_rm2.rename(columns={"rm2_category": "kategori"})
        sentiment_by_rm2 = sentiment_by_rm2[sentiment_by_rm2["kategori"] != "Tidak teridentifikasi"].copy()
        display_dataframe(sentiment_by_rm2)

        available = [c for c in LABEL_ORDER if c in sentiment_by_rm2.columns]
        if available and PLOTLY_AVAILABLE:
            long_df = sentiment_by_rm2.melt(id_vars="kategori", value_vars=available, var_name="Sentimen", value_name="Jumlah")
            fig = px.bar(long_df, x="Jumlah", y="kategori", color="Sentimen", orientation="h", color_discrete_map=LABEL_COLORS, title="Sentimen pada Respons Uang Tunai Teridentifikasi")
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(apply_dark_plot_layout(fig, 560), use_container_width=True)
    else:
        st.info("Tabel hubungan sentimen dengan Bentuk Kekhawatiran & Tanggapan belum tersedia.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Contoh Komentar per Bentuk Kekhawatiran & Tanggapan")
    render_rm2_examples()

    with st.expander("Lihat distribusi semua kategori termasuk Tidak teridentifikasi", expanded=False):
        display_dataframe(rm2_all)
        if not rm2_all.empty:
            plot_horizontal_bar(rm2_all.sort_values("jumlah", ascending=True), "jumlah", "kategori", "Semua Respons Uang Tunai", "#64748B", 520)


def tab_rm3(eval_df, best):
    render_page_header("Perbandingan Performa Model", "Membandingkan Naive Bayes, SVM, dan IndoBERT berdasarkan metrik evaluasi utama.")
    render_model_cards(eval_df, best)

    st.subheader("Grafik Perbandingan Model")
    plot_model_metrics(eval_df)

    with st.expander("Lihat heatmap performa model", expanded=False):
        plot_model_heatmap(eval_df)

    with st.expander("Lihat tabel evaluasi lengkap", expanded=False):
        display_dataframe(eval_df)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Classification Report dan Confusion Matrix")
    model_names = list(MODEL_CONFIG.keys())
    selected_index = model_names.index(str(best["model"])) if str(best["model"]) in model_names else model_names.index("IndoBERT")
    selected_model = st.selectbox("Pilih model", model_names, index=selected_index)

    cfg = MODEL_CONFIG[selected_model]
    report_df, _ = load_first_existing(cfg["report_files"], f"Classification report {selected_model}")
    cm_path = find_existing_path(cfg["confusion_matrix_files"])

    col_report, col_cm = st.columns([1.25, 1])
    with col_report:
        st.markdown(f"**Classification Report - {selected_model}**")
        if not report_df.empty:
            display_dataframe(clean_unnamed_columns(report_df))
        else:
            st.info("Classification report belum tersedia.")
    with col_cm:
        st.markdown(f"**Confusion Matrix - {selected_model}**")
        if cm_path is not None:
            show_image_or_warning(cm_path, f"Confusion Matrix {selected_model}")
        else:
            st.info("Gambar confusion matrix belum tersedia.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("McNemar Test")
    mcnemar_df = load_analysis_table("mcnemar_results_summary")
    if mcnemar_df.empty:
        mcnemar_df, _ = load_first_existing([RESULT_FILES["mcnemar_results"], RESULT_FILES["mcnemar_results_alt"]], "mcnemar_results")
    display_dataframe(mcnemar_df)

    if not mcnemar_df.empty and "p_value" in mcnemar_df.columns:
        with st.expander("Interpretasi McNemar Test", expanded=False):
            for _, row in mcnemar_df.iterrows():
                a = row.get("model_a", row.get("Model_A", "Model A"))
                b = row.get("model_b", row.get("Model_B", "Model B"))
                try:
                    p = float(row.get("p_value"))
                except Exception:
                    continue
                if p < 0.05:
                    st.write(f"- **{a} vs {b}**: p-value = {p:.6f}, terdapat perbedaan performa yang signifikan.")
                else:
                    st.write(f"- **{a} vs {b}**: p-value = {p:.6f}, tidak terdapat perbedaan performa yang signifikan.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Ringkasan Error Analysis Model Terbaik")
    error_pair = load_analysis_table("error_pair_best_model")
    error_analysis = load_analysis_table("error_analysis_best_model")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Jenis Kesalahan Teratas**")
        display_dataframe(error_pair.head(10) if not error_pair.empty else error_pair)
    with c2:
        st.markdown("**Contoh Kesalahan Prediksi**")
        display_dataframe(error_analysis.head(10) if not error_analysis.empty else error_analysis)


def render_prediction_result(label, model_name, preprocessing_desc, interpretation):
    label = normalize_label(label)
    class_name = {"Negatif": "pred-negatif", "Netral": "pred-netral", "Positif": "pred-positif"}.get(label, "pred-netral")
    st.markdown(
        f"""
<div class="prediction-card">
    <div class="prediction-label {class_name}">{html.escape(label)}</div>
    <div><strong>Model:</strong> {html.escape(model_name)}</div>
    <div><strong>Jenis preprocessing:</strong> {html.escape(preprocessing_desc)}</div>
    <div class="divider"></div>
    <div>{html.escape(interpretation)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def tab_simulation(best):
    render_page_header("Simulasi Prediksi & Kesimpulan Penelitian", "Gunakan model yang tersedia untuk memprediksi sentimen komentar baru dan membaca ringkasan kesimpulan penelitian.")
    st.subheader("A. Simulasi Prediksi Komentar Baru")
    model_names = list(MODEL_CONFIG.keys())
    default_index = model_names.index("SVM") if "SVM" in model_names else 0
    model_name = st.selectbox("Pilih model prediksi", model_names, index=default_index)
    user_text = st.text_area("Masukkan komentar baru", placeholder="Contoh: Saya takut semua transaksi nanti dikontrol pemerintah.", height=135)

    if st.button("Prediksi Sentimen", type="primary"):
        if not user_text.strip():
            st.warning("Masukkan komentar terlebih dahulu.")
        else:
            cfg = MODEL_CONFIG[model_name]
            preprocessing_desc = get_preprocessing_description(model_name, cfg["preprocessing"])
            model_path = find_existing_path(cfg["model_path_candidates"])
            if model_path is None:
                st.error("File/folder model belum ditemukan. Cek kembali folder Models.")
            elif cfg["type"] == "sklearn_pipeline":
                model, err = load_sklearn_pipeline(str(model_path))
                if err:
                    st.error(err)
                else:
                    try:
                        label, processed = predict_with_sklearn_pipeline(model, user_text)
                        render_prediction_result(label, model_name, preprocessing_desc, get_prediction_interpretation(label))
                        with st.expander("Lihat teks setelah preprocessing", expanded=False):
                            st.code(processed)
                    except Exception as exc:
                        st.error(f"Prediksi gagal dijalankan. Detail: {exc}")
            else:
                with st.spinner("Memuat IndoBERT dan melakukan prediksi..."):
                    tokenizer, model, device, err = load_indobert_model(str(model_path))
                if err:
                    st.error(err)
                else:
                    try:
                        label, processed = predict_with_indobert(tokenizer, model, device, user_text)
                        render_prediction_result(label, model_name, preprocessing_desc, get_prediction_interpretation(label))
                        with st.expander("Lihat teks setelah light cleaning", expanded=False):
                            st.code(processed)
                            st.caption(f"Device IndoBERT: {device}")
                    except Exception as exc:
                        st.error(f"Prediksi IndoBERT gagal dijalankan. Detail: {exc}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("B. Kesimpulan Penelitian")
    rq_summary = load_analysis_table("research_questions_summary")
    if not rq_summary.empty:
        display_dataframe(rq_summary)
    else:
        render_info_box(
            f"<strong>RM1.</strong> Dashboard menampilkan distribusi sentimen akhir pengguna YouTube.<br><br>"
            f"<strong>RM2.</strong> Analisis tanggapan uang tunai difokuskan pada kategori teridentifikasi.<br><br>"
            f"<strong>RM3.</strong> Model terbaik adalah <strong>{best['model']}</strong> berdasarkan <strong>{best['metric']}</strong>."
        )

def main():
    labeled_df = load_labeled_data()
    sentiment_dist = make_sentiment_distribution(labeled_df)
    eval_df = make_model_comparison()
    best = make_best_model_summary(eval_df)
    rm2_all = load_rm2_distribution(identified_only=False)
    rm2_identified = load_rm2_distribution(identified_only=True)
    total_comments = int(len(labeled_df)) if not labeled_df.empty else int(sentiment_dist["jumlah"].sum())

    menu = build_sidebar(best, total_comments)

    if menu == "Overview":
        tab_overview(labeled_df, sentiment_dist, eval_df, rm2_identified, best)
    elif menu == "Distribusi Sentimen":
        tab_rm1(labeled_df, sentiment_dist)
    elif menu == "Bentuk Kekhawatiran & Tanggapan":
        tab_rm2(rm2_all, rm2_identified)
    elif menu == "Perbandingan Model":
        tab_rm3(eval_df, best)
    elif menu == "Simulasi":
        tab_simulation(best)

    st.markdown(
        '<div class="small-caption">Dashboard ini membaca file hasil penelitian yang sudah diproses. Tidak ada scraping ulang, labelling ulang, preprocessing ulang dataset besar, atau training ulang model.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
