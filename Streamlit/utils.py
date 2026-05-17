from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import LABEL_NORMALIZATION, LABEL_ORDER, resolve_project_path


NORMALIZATION_MAP = {
    "ga": "tidak", "gak": "tidak", "gk": "tidak", "nggak": "tidak",
    "enggak": "tidak", "tdk": "tidak", "yg": "yang", "dgn": "dengan",
    "utk": "untuk", "krn": "karena", "karna": "karena", "duit": "uang",
    "rp": "rupiah", "rph": "rupiah", "bi": "bank indonesia",
    "cbdc": "central bank digital currency", "jd": "jadi", "jdi": "jadi",
    "sm": "sama", "trs": "terus", "trus": "terus", "bgt": "banget",
    "org": "orang", "pake": "pakai", "make": "pakai", "klo": "kalau",
    "kalo": "kalau", "dr": "dari", "bs": "bisa", "sdh": "sudah",
    "udh": "sudah", "udah": "sudah", "blm": "belum", "belom": "belum",
    "knp": "kenapa", "ap": "apa", "lg": "lagi", "tp": "tapi",
    "sy": "saya", "gw": "saya", "gue": "saya", "gua": "saya",
    "lu": "kamu", "lo": "kamu", "km": "kamu",
}

KEEP_SENTIMENT_WORDS = {
    "tidak", "bukan", "jangan", "belum", "tanpa", "kurang", "takut",
    "aman", "bahaya", "mahal", "mudah", "sulit", "baik", "buruk",
    "setuju", "tolak", "percaya", "curiga", "risiko", "untung", "rugi",
}


def normalize_label(label: Any) -> str:
    if pd.isna(label):
        return "Netral"
    if label in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[label]
    text = str(label).strip()
    if text in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[text]
    lower = text.lower()
    return LABEL_NORMALIZATION.get(lower, text)


def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_label)


@st.cache_data(show_spinner=False)
def read_csv_cached(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def read_excel_cached(path_str: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    return pd.read_excel(path_str, sheet_name=sheet_name)


def clean_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if not unnamed:
        return df

    first = unnamed[0]
    values = df[first].dropna().astype(str).tolist()
    looks_like_index = all(v.isdigit() for v in values[:10]) if values else True

    if not looks_like_index:
        df = df.rename(columns={first: "label"})
    else:
        df = df.drop(columns=[first])

    other_unnamed = [c for c in unnamed[1:] if c in df.columns]
    if other_unnamed:
        df = df.drop(columns=other_unnamed)
    return df


def load_table_safely(path: Path, label: str = "file") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = read_excel_cached(str(path))
        else:
            df = read_csv_cached(str(path))
        return clean_unnamed_columns(df)
    except Exception as exc:
        st.warning(f"{label} gagal dibaca: `{path}`. Detail: {exc}")
        return pd.DataFrame()


def load_first_existing(paths: list[Path], label: str = "file") -> tuple[pd.DataFrame, Path | None]:
    for path in paths:
        if path.exists():
            return load_table_safely(path, label), path
    return pd.DataFrame(), None


def read_excel_sheets_safely(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    try:
        sheets = pd.read_excel(path, sheet_name=None)
        return {name: clean_unnamed_columns(df) for name, df in sheets.items()}
    except Exception as exc:
        st.warning(f"File Excel gagal dibaca: `{path}`. Detail: {exc}")
        return {}


def find_existing_path(paths: list[str | Path]) -> Path | None:
    for path in paths:
        resolved = resolve_project_path(path)
        if resolved.exists():
            return resolved
    return None


def show_image_or_warning(path: Path, caption: str | None = None) -> None:
    import base64
    import mimetypes

    if not path.exists() or not path.is_file():
        st.info(f"Gambar belum tersedia: `{path}`")
        return

    try:
        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        image_base64 = base64.b64encode(path.read_bytes()).decode("utf-8")

        caption_html = ""
        if caption:
            caption_html = (
                f"<div style='text-align:center; color:#CBD5E1; "
                f"font-size:0.85rem; margin-top:0.35rem;'>{caption}</div>"
            )

        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="data:{mime_type};base64,{image_base64}"
                     style="max-width:100%; border-radius:12px;" />
                {caption_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.warning(f"Gambar gagal ditampilkan: `{path}`. Detail: {exc}")


def format_metric_value(value: Any, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, (float, np.floating)):
        return f"{value:.{decimals}f}"
    return str(value)


def format_percent(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}%"
    except (TypeError, ValueError):
        return "-"


def clean_text_basic(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_tokens(text: str) -> str:
    return " ".join(NORMALIZATION_MAP.get(token, token) for token in text.split())


@st.cache_resource(show_spinner=False)
def load_sastrawi_components() -> dict[str, Any]:
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

        stemmer = StemmerFactory().create_stemmer()
        stopwords = set(StopWordRemoverFactory().get_stop_words()) - KEEP_SENTIMENT_WORDS
        return {"available": True, "stemmer": stemmer, "stopwords": stopwords}
    except ImportError:
        return {"available": False, "stemmer": None, "stopwords": set()}


def preprocess_for_nb_svm(text: str) -> str:
    cleaned = normalize_tokens(clean_text_basic(text))
    sastrawi = load_sastrawi_components()

    if not sastrawi["available"]:
        return cleaned or "emptytext"

    tokens = [token for token in cleaned.split() if token not in sastrawi["stopwords"]]
    cleaned = " ".join(tokens).strip()
    if cleaned:
        cleaned = sastrawi["stemmer"].stem(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip() or "emptytext"


def preprocess_for_indobert(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_tokens(text)
    return re.sub(r"\s+", " ", text).strip() or "emptytext"


@st.cache_resource(show_spinner=False)
def load_sklearn_pipeline(model_path_str: str):
    model_path = resolve_project_path(model_path_str)
    if not model_path.exists():
        return None, f"Model tidak ditemukan: {model_path}"
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as exc:
        return None, f"Model gagal dimuat: {model_path}. Detail: {exc}"


@st.cache_resource(show_spinner=False)
def load_indobert_model(model_path_str: str):
    model_path = resolve_project_path(model_path_str)
    if not model_path.exists():
        return None, None, None, f"Folder model IndoBERT tidak ditemukan: {model_path}"

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None, None, None, "Library transformers/torch belum tersedia."

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model.to(device)
        model.eval()
        return tokenizer, model, device, None
    except Exception as exc:
        return None, None, None, f"Model IndoBERT gagal dimuat dari {model_path}. Detail: {exc}"


def predict_with_sklearn_pipeline(model, text: str) -> tuple[str, str]:
    processed_text = preprocess_for_nb_svm(text)
    prediction = model.predict([processed_text])[0]
    return normalize_label(prediction), processed_text


def _resolve_indobert_label(model, pred_id: int) -> str:
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict):
        label = id2label.get(pred_id) or id2label.get(str(pred_id))
        if label is not None:
            return normalize_label(label)
    if pred_id < len(LABEL_ORDER):
        return LABEL_ORDER[pred_id]
    return normalize_label(pred_id)


def predict_with_indobert(tokenizer, model, device: str, text: str) -> tuple[str, str]:
    import torch

    processed_text = preprocess_for_indobert(text)
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = int(torch.argmax(outputs.logits, dim=1).item())

    return _resolve_indobert_label(model, pred_id), processed_text


def get_prediction_interpretation(label: str) -> str:
    label = normalize_label(label)
    interpretations = {
        "Negatif": "Komentar ini diprediksi bernada negatif, menunjukkan kekhawatiran, penolakan, kritik, atau ketidakpercayaan.",
        "Netral": "Komentar ini diprediksi netral, cenderung informatif, bertanya, atau tidak menunjukkan dukungan maupun penolakan yang jelas.",
        "Positif": "Komentar ini diprediksi positif, menunjukkan dukungan, penerimaan, atau pandangan optimis terhadap digitalisasi pembayaran.",
    }
    return interpretations.get(label, "Label prediksi tidak dikenali.")


def get_preprocessing_description(model_name: str, preprocessing_key: str) -> str:
    if preprocessing_key == "nb_svm":
        return (
            "Basic cleaning, normalisasi kata tidak baku, stopword removal, dan stemming jika Sastrawi tersedia. "
            f"Input kemudian diproses oleh pipeline {model_name}."
        )
    if preprocessing_key == "indobert":
        return "Light cleaning tanpa stopword removal dan tanpa stemming, lalu tokenisasi menggunakan tokenizer IndoBERT."
    return "Preprocessing tidak dikenali."
