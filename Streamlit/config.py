from __future__ import annotations
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "Processed"
ANALYSIS_DIR = DATA_DIR / "Analysis_Result"
ANALYSIS_TABLE_DIR = ANALYSIS_DIR / "Tables"
ANALYSIS_FIGURE_DIR = ANALYSIS_DIR / "Figures"
ANALYSIS_EXAMPLE_DIR = ANALYSIS_DIR / "Examples"

MODELING_RESULTS_DIR = DATA_DIR / "Modeling_Results"
MODELLING_RESULTS_DIR_ALT = DATA_DIR / "Modelling_Results"
MODEL_DIR = PROJECT_ROOT / "Models"

APP_TITLE = "Dashboard Analisis Sentimen Rupiah Digital"
APP_SUBTITLE = (
    "Ringkasan hasil analisis komentar YouTube terhadap Rupiah Digital, "
    "keberlanjutan uang tunai, dan performa model klasifikasi sentimen."
)

LABEL_ORDER = ["Negatif", "Netral", "Positif"]

LABEL_NORMALIZATION = {
    "negative": "Negatif",
    "negatif": "Negatif",
    "neutral": "Netral",
    "netral": "Netral",
    "positive": "Positif",
    "positif": "Positif",
    "LABEL_0": "Negatif",
    "LABEL_1": "Netral",
    "LABEL_2": "Positif",
    0: "Negatif",
    1: "Netral",
    2: "Positif",
}

LABEL_COLORS = {
    "Negatif": "#EF4444",
    "Netral": "#94A3B8",
    "Positif": "#22C55E",
}

THEME = {
    "page_bg": "#0B1120",
    "card_bg": "#111827",
    "sidebar_bg": "#0F172A",
    "primary": "#2563EB",
    "secondary": "#38BDF8",
    "text": "#F8FAFC",
    "muted": "#CBD5E1",
    "border": "#334155",
    "table_header": "#1E293B",
    "table_row": "#111827",
    "plot_bg": "#111827",
}

RM2_CATEGORY_ORDER = [
    "Kekhawatiran uang tunai dihapus",
    "Kebutuhan terhadap uang tunai",
    "Akses teknologi dan infrastruktur",
    "Keamanan data dan kontrol",
    "Kepercayaan terhadap regulator",
    "Efisiensi dan kemudahan transaksi",
    "Tanggapan mendukung digitalisasi",
    "Literasi dan kesiapan masyarakat",
    "Tidak teridentifikasi",
]

RESULT_FILES = {
    "labeled_data": PROCESSED_DIR / "04_hasil_labeling_final_lexicon.csv",
    "final_sentiment_distribution": ANALYSIS_TABLE_DIR / "final_sentiment_distribution.csv",
    "cash_response_distribution": ANALYSIS_TABLE_DIR / "cash_response_distribution_refined.csv",
    "cash_response_distribution_without_unidentified": ANALYSIS_TABLE_DIR / "cash_response_distribution_refined_without_unidentified.csv",
    "sentiment_by_cash_category": ANALYSIS_TABLE_DIR / "sentiment_by_cash_category_refined.csv",
    "model_comparison_summary": ANALYSIS_TABLE_DIR / "model_comparison_summary.csv",
    "best_model_summary_analysis": ANALYSIS_TABLE_DIR / "best_model_summary_analysis.csv",
    "mcnemar_results_summary": ANALYSIS_TABLE_DIR / "mcnemar_results_summary.csv",
    "error_analysis_best_model": ANALYSIS_TABLE_DIR / "error_analysis_best_model.csv",
    "error_pair_best_model": ANALYSIS_TABLE_DIR / "error_pair_best_model.csv",
    "research_questions_summary": ANALYSIS_TABLE_DIR / "research_questions_summary.csv",
    "draft_narrative_bab_iv": ANALYSIS_TABLE_DIR / "draft_narrative_bab_iv.csv",
    "examples_by_sentiment": ANALYSIS_EXAMPLE_DIR / "example_comments_by_sentiment.xlsx",
    "examples_by_rm2": ANALYSIS_EXAMPLE_DIR / "example_comments_by_cash_category_refined.xlsx",
    "evaluation_results": MODELING_RESULTS_DIR / "evaluation_results.csv",
    "evaluation_results_alt": MODELLING_RESULTS_DIR_ALT / "evaluation_results.csv",
    "mcnemar_results": MODELING_RESULTS_DIR / "mcnemar_results.csv",
    "mcnemar_results_alt": MODELLING_RESULTS_DIR_ALT / "mcnemar_results.csv",
}

MODEL_CONFIG = {
    "Naive Bayes": {
        "type": "sklearn_pipeline",
        "model_path_candidates": [
            "Models/naive_bayes_tuned_pipeline.pkl",
            "Models/naive_bayes_pipeline.pkl",
            "Models/naive_bayes_model.pkl",
        ],
        "preprocessing": "nb_svm",
        "report_files": [
            MODELING_RESULTS_DIR / "classification_report_naive_bayes.csv",
            MODELLING_RESULTS_DIR_ALT / "classification_report_naive_bayes.csv",
            ANALYSIS_TABLE_DIR / "classification_report_naive_bayes.csv",
        ],
        "confusion_matrix_files": [
            ANALYSIS_FIGURE_DIR / "confusion_matrix_naive_bayes.png",
            MODELING_RESULTS_DIR / "Figures" / "confusion_matrix_naive_bayes.png",
            MODELLING_RESULTS_DIR_ALT / "Figures" / "confusion_matrix_naive_bayes.png",
        ],
    },
    "SVM": {
        "type": "sklearn_pipeline",
        "model_path_candidates": [
            "Models/svm_tuned_pipeline.pkl",
            "Models/svm_pipeline.pkl",
            "Models/svm_model.pkl",
        ],
        "preprocessing": "nb_svm",
        "report_files": [
            MODELING_RESULTS_DIR / "classification_report_svm.csv",
            MODELLING_RESULTS_DIR_ALT / "classification_report_svm.csv",
            ANALYSIS_TABLE_DIR / "classification_report_svm.csv",
        ],
        "confusion_matrix_files": [
            ANALYSIS_FIGURE_DIR / "confusion_matrix_svm.png",
            MODELING_RESULTS_DIR / "Figures" / "confusion_matrix_svm.png",
            MODELLING_RESULTS_DIR_ALT / "Figures" / "confusion_matrix_svm.png",
        ],
    },
    "IndoBERT": {
        "type": "transformer",
        "model_path_candidates": [
            "Models/indobert_sentiment_model",
            "Models/indobert_model",
            "Models/IndoBERT",
        ],
        "preprocessing": "indobert",
        "report_files": [
            MODELING_RESULTS_DIR / "classification_report_indobert.csv",
            MODELLING_RESULTS_DIR_ALT / "classification_report_indobert.csv",
            ANALYSIS_TABLE_DIR / "classification_report_indobert.csv",
        ],
        "confusion_matrix_files": [
            ANALYSIS_FIGURE_DIR / "confusion_matrix_indobert.png",
            MODELING_RESULTS_DIR / "Figures" / "confusion_matrix_indobert.png",
            MODELLING_RESULTS_DIR_ALT / "Figures" / "confusion_matrix_indobert.png",
        ],
    },
}

def resolve_project_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path
