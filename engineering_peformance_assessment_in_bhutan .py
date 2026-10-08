"""
AI-Based Assessment of Civil Engineering Practice and Project Performance in Bhutan

GitHub setup:
1. Save this file as app.py.
2. Create requirements.txt containing:
   streamlit
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
3. Run locally with: streamlit run app.py

Important:
This classroom/research prototype uses synthetic project records only.
It is not an official assessment of Bhutanese engineers, contractors,
agencies, or individual projects. It is an educational decision-support
demonstration and should not be used to assign blame or make disciplinary
decisions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Engineering Performance in Bhutan",
    page_icon="🏗️",
    layout="wide",
)

TARGET = "performance_assessment"
POSITIVE = "Attention Required"


@st.cache_data
def generate_synthetic_data(n_rows=1000, seed=42):
    """Create fictional project records for educational use."""
    rng = np.random.default_rng(seed)

    dzongkhags = [
        "Thimphu", "Paro", "Punakha", "Wangdue Phodrang", "Chukha",
        "Sarpang", "Samtse", "Mongar", "Trashigang", "Bumthang",
        "Tsirang", "Dagana", "Trongsa", "Zhemgang", "Lhuentse",
        "Pemagatshel", "Samdrup Jongkhar", "Haa", "Gasa", "Tashi Yangtse",
    ]

    project_types = [
        "Building",
        "Road",
        "Water & Sanitation",
        "Electrical",
        "Maintenance/Renovation",
    ]

    site_access = ["Easy", "Moderate", "Difficult"]
    material_availability = ["High", "Moderate", "Low"]
    weather_impact = ["Low", "Moderate", "High"]

    dzongkhag = rng.choice(dzongkhags, n_rows)
    project_type = rng.choice(project_types, n_rows)
    contract_amount = np.clip(
        rng.lognormal(mean=np.log(4.5), sigma=0.8, size=n_rows), 0.5, 50
    ).round(2)
    contract_duration = rng.integers(3, 25, n_rows)
    contractor_experience = np.clip(
        rng.normal(8, 4, n_rows), 1, 25
    ).round(1)
    work_progress = np.clip(
        rng.normal(70, 22, n_rows), 5, 100
    ).round(1)
    financial_progress = np.clip(
        work_progress + rng.normal(0, 12, n_rows), 0, 100
    ).round(1)
    time_elapsed = np.clip(
        rng.normal(72, 22, n_rows), 5, 110
    ).round(1)
    variation_percent = np.clip(
        rng.gamma(shape=2.0, scale=2.2, size=n_rows), 0, 20
    ).round(1)
    extension_days = np.clip(
        rng.normal(18, 25, n_rows), 0, 120
    ).round().astype(int)
    manpower = np.clip(
        rng.normal(18, 9, n_rows), 3, 60
    ).round().astype(int)
    material = rng.choice(
        material_availability, n_rows, p=[0.45, 0.40, 0.15]
    )
    accessibility = rng.choice(
        site_access, n_rows, p=[0.35, 0.45, 0.20]
    )
    weather = rng.choice(
        weather_impact, n_rows, p=[0.50, 0.35, 0.15]
    )
    payment_delay = np.clip(
        rng.normal(18, 18, n_rows), 0, 100
    ).round().astype(int)

    quality_control = np.clip(
        rng.normal(7.0, 1.7, n_rows), 1, 10
    ).round(1)
    site_supervision = np.clip(
        rng.normal(7.0, 1.7, n_rows), 1, 10
    ).round(1)
    drawing_compliance = np.clip(
        rng.normal(7.5, 1.5, n_rows), 1, 10
    ).round(1)
    safety_practice = np.clip(
        rng.normal(7.2, 1.7, n_rows), 1, 10
    ).round(1)
    documentation = np.clip(
        rng.normal(7.0, 1.8, n_rows), 1, 10
    ).round(1)
    maintenance_planning = np.clip(
        rng.normal(6.5, 1.9, n_rows), 1, 10
    ).round(1)
    defect_rate = np.clip(
        rng.gamma(shape=1.8, scale=1.5, size=n_rows), 0, 12
    ).round(1)

    # Transparent fictional rule used only to create the teaching label.
    risk_score = (
        0.10 * variation_percent
        + 0.035 * extension_days
        + 0.025 * payment_delay
        + 0.20 * defect_rate
        + 0.45 * (10 - quality_control)
        + 0.40 * (10 - site_supervision)
        + 0.35 * (10 - drawing_compliance)
        + 0.35 * (10 - safety_practice)
        + 0.25 * (10 - documentation)
        + 0.20 * (10 - maintenance_planning)
        + 0.30 * (time_elapsed / 10)
        - 0.02 * contractor_experience
        + rng.normal(0, 1.2, n_rows)
    )

    threshold = np.quantile(risk_score, 0.65)

    performance_assessment = np.where(
        risk_score >= threshold,
        POSITIVE,
        "Generally Acceptable",
    )

    return pd.DataFrame(
        {
            "dzongkhag": dzongkhag,
            "project_type": project_type,
            "contract_amount_nu_million": contract_amount,
            "contract_duration_months": contract_duration,
            "contractor_experience_years": contractor_experience,
            "work_progress_percent": work_progress,
            "financial_progress_percent": financial_progress,
            "time_elapsed_percent": time_elapsed,
            "variation_percent": variation_percent,
            "extension_days": extension_days,
            "manpower": manpower,
            "material_availability": material,
            "site_accessibility": accessibility,
            "weather_impact": weather,
            "payment_delay_days": payment_delay,
            "quality_control_score": quality_control,
            "site_supervision_score": site_supervision,
            "drawing_compliance_score": drawing_compliance,
            "safety_practice_score": safety_practice,
            "documentation_score": documentation,
            "maintenance_planning_score": maintenance_planning,
            "defect_rate": defect_rate,
            TARGET: performance_assessment,
        }
    )


@st.cache_resource
def train_model(data):
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    categorical_columns = X.select_dtypes(include="object").columns.tolist()
    numeric_columns = X.select_dtypes(exclude="object").columns.tolist()

    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    positive_index = list(model.classes_).index(POSITIVE)
    probabilities = model.predict_proba(X_test)[:, positive_index]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(
            (y_test == POSITIVE).astype(int),
            probabilities,
        ),
        "matrix": confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_,
        ),
        "report": classification_report(
            y_test,
            predictions,
            output_dict=True,
        ),
        "classes": model.classes_,
    }

    return model, metrics


data = generate_synthetic_data()
model, metrics = train_model(data)

st.title("AI-Based Engineering Performance Assessment in Bhutan")
st.caption(
    "A beginner machine-learning project using fictional construction "
    "and engineering project records"
)

st.warning(
    "This prototype is not an official assessment of individual civil "
    "engineers, contractors, or government agencies. It should not be "
    "used for disciplinary, employment, procurement, or legal decisions. "
    "The current dataset is synthetic and does not represent actual "
    "Bhutanese project records."
)

overview_tab, data_tab, prediction_tab, results_tab = st.tabs(
    ["Overview", "Explore Data", "Project Assessment", "Model Results"]
)


with overview_tab:
    st.subheader("Project Objective")
    st.write(
        "This project demonstrates how machine learning can identify "
        "patterns in engineering project implementation that may be "
        "associated with quality, safety, time, cost, supervision, "
        "documentation, and long-term maintenance concerns in Bhutan."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Fictional project records", f"{len(data):,}")
    col2.metric("Input indicators", len(data.columns) - 1)
    col3.metric("Algorithm", "Logistic regression")

    st.markdown("### Key areas assessed")
    st.write(
        "• Construction quality and defects\n"
        "• Site supervision and drawing compliance\n"
        "• Safety practices\n"
        "• Project time and cost performance\n"
        "• Material availability and site accessibility\n"
        "• Documentation and quality control\n"
        "• Maintenance planning and sustainability"
    )

    st.markdown("### Machine-learning workflow")
    st.write(
        "1. Generate project data → 2. Split data → "
        "3. Preprocess variables → 4. Train model → "
        "5. Evaluate model → 6. Assess a project"
    )

    st.info(
        "A real Bhutanese study would require appropriately collected "
        "project records, clearly defined indicators, expert validation, "
        "ethical/data-governance review where applicable, and testing "
        "against real outcomes before the model could be used for "
        "engineering decision support."
    )


with data_tab:
    st.subheader("Exploratory Data Analysis")

    selected_feature = st.selectbox(
        "Choose an engineering indicator",
        [
            "variation_percent",
            "extension_days",
            "quality_control_score",
            "site_supervision_score",
            "drawing_compliance_score",
            "safety_practice_score",
            "documentation_score",
            "maintenance_planning_score",
            "defect_rate",
            "payment_delay_days",
        ],
    )

    figure, axis = plt.subplots(figsize=(9, 4.5))
    sns.histplot(
        data=data,
        x=selected_feature,
        hue=TARGET,
        multiple="stack",
        ax=axis,
    )
    axis.set_title(
        selected_feature.replace("_", " ").title()
        + " by Project Assessment"
    )
    st.pyplot(figure)

    st.markdown("### First 20 fictional project records")
    st.dataframe(data.head(20), use_container_width=True)

    st.download_button(
        "Download synthetic CSV",
        data.to_csv(index=False),
        "bhutan_engineering_project_synthetic.csv",
        "text/csv",
    )


with prediction_tab:
    st.subheader("Assess One Fictional Project")
    st.write(
        "Enter project characteristics below to demonstrate how the "
        "model identifies a project that may require additional attention."
    )

    left, right = st.columns(2)

    with left:
        dzongkhag = st.selectbox(
            "Dzongkhag",
            sorted(data["dzongkhag"].unique()),
        )
        project_type = st.selectbox(
            "Project type",
            sorted(data["project_type"].unique()),
        )
        contract_amount = st.slider(
            "Contract amount (Nu. million)",
            0.5, 50.0, 5.0, 0.5,
        )
        duration = st.slider(
            "Contract duration (months)",
            3, 24, 12,
        )
        experience = st.slider(
            "Contractor experience (years)",
            1.0, 25.0, 8.0, 0.5,
        )
        work_progress = st.slider(
            "Physical work progress (%)",
            0.0, 100.0, 70.0, 1.0,
        )
        financial_progress = st.slider(
            "Financial progress (%)",
            0.0, 100.0, 65.0, 1.0,
        )
        time_elapsed = st.slider(
            "Time elapsed (%)",
            0.0, 110.0, 70.0, 1.0,
        )
        variation = st.slider(
            "Variation (%)",
            0.0, 20.0, 3.0, 0.5,
        )

    with right:
        extension = st.slider(
            "Extension of time (days)",
            0, 120, 15,
        )
        manpower_input = st.slider(
            "Average manpower",
            3, 60, 18,
        )
        material = st.selectbox(
            "Material availability",
            ["High", "Moderate", "Low"],
        )
        accessibility = st.selectbox(
            "Site accessibility",
            ["Easy", "Moderate", "Difficult"],
        )
        weather = st.selectbox(
            "Weather impact",
            ["Low", "Moderate", "High"],
        )
        payment_delay = st.slider(
            "Payment delay (days)",
            0, 100, 15,
        )
        quality = st.slider(
            "Quality control score (1 low–10 high)",
            1.0, 10.0, 7.0, 0.5,
        )
        supervision = st.slider(
            "Site supervision score (1 low–10 high)",
            1.0, 10.0, 7.0, 0.5,
        )
        drawing = st.slider(
            "Drawing/specification compliance (1–10)",
            1.0, 10.0, 7.5, 0.5,
        )
        safety = st.slider(
            "Safety practice score (1–10)",
            1.0, 10.0, 7.0, 0.5,
        )
        documentation = st.slider(
            "Documentation score (1–10)",
            1.0, 10.0, 7.0, 0.5,
        )
        maintenance = st.slider(
            "Maintenance planning score (1–10)",
            1.0, 10.0, 6.5, 0.5,
        )
        defects = st.slider(
            "Observed defect rate",
            0.0, 12.0, 2.0, 0.5,
        )

    profile = pd.DataFrame(
        [
            {
                "dzongkhag": dzongkhag,
                "project_type": project_type,
                "contract_amount_nu_million": contract_amount,
                "contract_duration_months": duration,
                "contractor_experience_years": experience,
                "work_progress_percent": work_progress,
                "financial_progress_percent": financial_progress,
                "time_elapsed_percent": time_elapsed,
                "variation_percent": variation,
                "extension_days": extension,
                "manpower": manpower_input,
                "material_availability": material,
                "site_accessibility": accessibility,
                "weather_impact": weather,
                "payment_delay_days": payment_delay,
                "quality_control_score": quality,
                "site_supervision_score": supervision,
                "drawing_compliance_score": drawing,
                "safety_practice_score": safety,
                "documentation_score": documentation,
                "maintenance_planning_score": maintenance,
                "defect_rate": defects,
            }
        ]
    )

    if st.button("Run Project Assessment", type="primary"):
        positive_index = list(model.classes_).index(POSITIVE)
        probability = model.predict_proba(profile)[0][positive_index]

        st.metric(
            "Model-estimated probability of requiring additional attention",
            f"{probability:.1%}",
        )

        if probability >= 0.50:
            st.warning(
                "Demo result: the project shows patterns associated with "
                "higher attention requirements. This is a model exercise, "
                "not a finding of poor engineering practice."
            )
        else:
            st.success(
                "Demo result: the project shows fewer of the patterns "
                "associated with the attention-required class. This does "
                "not prove that the project is free from engineering risks."
            )


with results_tab:
    st.subheader("Hold-out Test Results")

    metric1, metric2 = st.columns(2)
    metric1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    metric2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        metrics["matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=metrics["classes"],
        yticklabels=metrics["classes"],
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("Confusion Matrix")
    st.pyplot(figure)

    st.markdown("### Classification Report")
    report_table = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(report_table.round(3), use_container_width=True)

    st.caption(
        "The random seed and train-test split are fixed for reproducibility. "
        "Because the demonstration target is generated using a fictional "
        "rule, these performance metrics must not be interpreted as evidence "
        "about actual engineering performance in Bhutan."
    )


st.divider()
st.caption(
    "Civil Engineering AI Project | Synthetic Data | Educational/Research Prototype"
)
