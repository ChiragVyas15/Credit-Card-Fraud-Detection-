from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "creditcard.csv"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071a2e 0%, #0d2947 100%);
    }
    [data-testid="stSidebar"] * { color: #f7fbff; }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #071a2e 0%, #0c4267 65%, #087e8b 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(7, 26, 46, .18);
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.55rem; letter-spacing: -.04em; }
    .hero p { margin: .55rem 0 0; color: #c9e8f1; font-size: 1.05rem; }
    .eyebrow {
        color: #65d6c7;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .result-card {
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-top: .5rem;
        border: 1px solid;
    }
    .safe { background: #ecfdf5; border-color: #6ee7b7; color: #065f46; }
    .danger { background: #fff1f2; border-color: #fda4af; color: #9f1239; }
    .result-card h2 { margin: 0 0 .35rem; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5eaf1;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(7, 26, 46, .04);
    }
    .note {
        background: #eef6ff;
        border-left: 4px solid #168aad;
        padding: .8rem 1rem;
        border-radius: 8px;
        color: #183b56;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Training fraud detection model...")
def train_model():
    data = load_data()
    legit = data[data["Class"] == 0].sample(
        n=int((data["Class"] == 1).sum()), random_state=42
    )
    fraud = data[data["Class"] == 1]
    balanced = pd.concat([legit, fraud]).sample(frac=1, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(
        balanced[FEATURES],
        balanced["Class"],
        test_size=0.2,
        stratify=balanced["Class"],
        random_state=2,
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "auc": roc_auc_score(y_test, probabilities),
        "confusion": confusion_matrix(y_test, predictions),
    }
    return model, metrics


def score_transactions(model, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = frame.copy()
    result["Fraud probability"] = model.predict_proba(result[FEATURES])[:, 1]
    result["Prediction"] = np.where(
        result["Fraud probability"] >= threshold, "Fraud", "Legitimate"
    )
    result["Risk score"] = (result["Fraud probability"] * 100).round(1)
    return result


try:
    data = load_data()
    model, model_metrics = train_model()
except Exception as exc:
    st.error(f"Unable to start the app: {exc}")
    st.stop()


with st.sidebar:
    st.markdown("## FraudShield AI")
    st.caption("Transaction intelligence console")
    st.divider()
    threshold = st.slider(
        "Fraud alert threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Lower values catch more suspicious transactions but may create more alerts.",
    )
    st.markdown("### Model")
    st.write("Logistic Regression")
    st.write("Standardized PCA features")
    st.write("Balanced training sample")
    st.divider()
    st.caption(
        "This app is an educational decision-support demo, not a replacement "
        "for bank-grade fraud controls."
    )


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Real-time risk intelligence</div>
      <h1>Credit Card Fraud Detection</h1>
      <p>Explore transaction patterns, estimate fraud risk, and screen CSV files in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, inspect_tab, batch_tab, model_tab = st.tabs(
    ["Overview", "Inspect transaction", "Batch screening", "Model insights"]
)

with overview_tab:
    total = len(data)
    fraud_count = int(data["Class"].sum())
    cols = st.columns(4)
    cols[0].metric("Transactions", f"{total:,}")
    cols[1].metric("Confirmed fraud", f"{fraud_count:,}")
    cols[2].metric("Fraud rate", f"{fraud_count / total:.3%}")
    cols[3].metric("Test ROC-AUC", f"{model_metrics['auc']:.1%}")

    st.subheader("Dataset snapshot")
    chart_data = pd.DataFrame(
        {
            "Transaction type": ["Legitimate", "Fraud"],
            "Count": [total - fraud_count, fraud_count],
        }
    ).set_index("Transaction type")
    left, right = st.columns([1.15, 1])
    with left:
        st.bar_chart(chart_data)
    with right:
        st.markdown("#### What the model sees")
        st.write(
            "`V1` through `V28` are anonymized principal components. `Time` is "
            "seconds since the first transaction, and `Amount` is the purchase value."
        )
        st.markdown(
            '<div class="note">Fraud is extremely rare in the original dataset, '
            "so accuracy alone can be misleading. Recall and ROC-AUC are included "
            "in the model panel.</div>",
            unsafe_allow_html=True,
        )

with inspect_tab:
    st.subheader("Inspect one transaction")
    st.write("Start with a real dataset example, then adjust any feature before scoring.")

    source_col, sample_col = st.columns([1, 2])
    with source_col:
        sample_type = st.radio(
            "Example source", ["Legitimate", "Fraud"], horizontal=True
        )
    sample_pool = data[data["Class"] == (1 if sample_type == "Fraud" else 0)]
    with sample_col:
        sample_index = st.selectbox(
            "Dataset row",
            sample_pool.index,
            format_func=lambda index: (
                f"Row {index:,}  |  Amount {sample_pool.loc[index, 'Amount']:,.2f}"
            ),
        )

    sample = sample_pool.loc[sample_index, FEATURES]
    with st.form("transaction_form"):
        basic_left, basic_right = st.columns(2)
        time_value = basic_left.number_input(
            "Time (seconds)", min_value=0.0, value=float(sample["Time"])
        )
        amount_value = basic_right.number_input(
            "Transaction amount", min_value=0.0, value=float(sample["Amount"])
        )

        feature_values = {}
        with st.expander("Advanced PCA features (V1-V28)", expanded=False):
            feature_columns = st.columns(4)
            for position, feature in enumerate(FEATURES[1:-1]):
                feature_values[feature] = feature_columns[position % 4].number_input(
                    feature,
                    value=float(sample[feature]),
                    format="%.6f",
                )
        submitted = st.form_submit_button(
            "Analyze transaction", type="primary", use_container_width=True
        )

    if submitted:
        transaction = {"Time": time_value, **feature_values, "Amount": amount_value}
        transaction_frame = pd.DataFrame([transaction], columns=FEATURES)
        probability = float(model.predict_proba(transaction_frame)[0, 1])
        is_fraud = probability >= threshold
        style = "danger" if is_fraud else "safe"
        title = "High-risk transaction" if is_fraud else "Likely legitimate"
        detail = (
            "This transaction crosses the current alert threshold."
            if is_fraud
            else "This transaction remains below the current alert threshold."
        )
        st.markdown(
            f"""
            <div class="result-card {style}">
              <h2>{title}</h2>
              <strong>Risk score: {probability:.1%}</strong>
              <p>{detail} Threshold: {threshold:.0%}.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(probability, text=f"Estimated fraud probability: {probability:.1%}")

with batch_tab:
    st.subheader("Screen a transaction file")
    st.write(
        "Upload a CSV containing `Time`, `V1` through `V28`, and `Amount`. "
        "A `Class` column is optional."
    )
    template = data[FEATURES].head(5).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV template",
        template,
        file_name="fraud_screening_template.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Upload transactions", type="csv")

    if upload is not None:
        try:
            uploaded_data = pd.read_csv(upload)
            missing = [feature for feature in FEATURES if feature not in uploaded_data]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                scored = score_transactions(model, uploaded_data, threshold)
                flagged = int((scored["Prediction"] == "Fraud").sum())
                summary = st.columns(3)
                summary[0].metric("Rows screened", f"{len(scored):,}")
                summary[1].metric("Alerts raised", f"{flagged:,}")
                summary[2].metric(
                    "Alert rate", f"{flagged / len(scored):.1%}" if len(scored) else "0%"
                )
                st.dataframe(
                    scored.sort_values("Fraud probability", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Fraud probability": st.column_config.ProgressColumn(
                            "Fraud probability", min_value=0, max_value=1, format="%.1%"
                        )
                    },
                )
                st.download_button(
                    "Download scored results",
                    scored.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_screening_results.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as exc:
            st.error(f"Could not process this file: {exc}")

with model_tab:
    st.subheader("Model performance")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{model_metrics['accuracy']:.1%}")
    metric_cols[1].metric("Precision", f"{model_metrics['precision']:.1%}")
    metric_cols[2].metric("Recall", f"{model_metrics['recall']:.1%}")
    metric_cols[3].metric("ROC-AUC", f"{model_metrics['auc']:.1%}")

    confusion = model_metrics["confusion"]
    st.markdown("#### Confusion matrix")
    st.dataframe(
        pd.DataFrame(
            confusion,
            index=["Actual legitimate", "Actual fraud"],
            columns=["Predicted legitimate", "Predicted fraud"],
        ),
        use_container_width=True,
    )
    st.caption(
        "Metrics are calculated on a stratified 20% holdout from a reproducible "
        "balanced sample. Results may differ from performance in live banking traffic."
    )
