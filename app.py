import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="Cardiovascular Risk Screening",
    page_icon="❤️",
    layout="centered"
)

# ============================================================
# 样式：背景 + 卡片 + 按钮
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #eef4ff 0%, #f2fbf7 100%);
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #143d59;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #4a5568;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .info-card {
        background: white;
        padding: 1.2rem 1.2rem 0.8rem 1.2rem;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(20, 61, 89, 0.08);
        margin-bottom: 1rem;
    }

    .result-card {
        background: white;
        padding: 1.3rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(20, 61, 89, 0.10);
        border-left: 8px solid #2b6cb0;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .risk-low {
        color: #2f855a;
        font-weight: 800;
    }

    .risk-moderate {
        color: #b7791f;
        font-weight: 800;
    }

    .risk-high {
        color: #dd6b20;
        font-weight: 800;
    }

    .risk-very-high {
        color: #c53030;
        font-weight: 800;
    }

    div.stButton > button {
        background-color: #2b6cb0;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1.2rem;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background-color: #1f4e85;
        color: white;
    }

    .small-note {
        color: #718096;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 风险分层
# ============================================================
def risk_level_from_probability(prob: float) -> str:
    if prob < 0.03:
        return "Low"
    elif prob < 0.08:
        return "Moderate"
    elif prob < 0.15:
        return "High"
    else:
        return "Very High"


def risk_level_class(level: str) -> str:
    mapping = {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "High": "risk-high",
        "Very High": "risk-very-high"
    }
    return mapping.get(level, "")

# ============================================================
# 模型加载
# ============================================================
@st.cache_resource
def load_user_model():
    model = joblib.load("calibrated_user_model.joblib")
    threshold = joblib.load("calibrated_user_threshold.joblib")["threshold"]
    return model, threshold

# ============================================================
# 本地预测
# ============================================================
def predict_user_local(payload: dict):
    model, threshold = load_user_model()
    sample_df = pd.DataFrame([payload])

    prob = model.predict_proba(sample_df)[:, 1][0]
    pred = int(prob >= threshold)
    level = risk_level_from_probability(prob)

    return {
        "risk_probability": float(prob),
        "risk_level": level,
        "risk_prediction": pred,
        "used_threshold": float(threshold),
        "message": "This tool is for cardiovascular risk screening only and does not constitute a medical diagnosis."
    }

# ============================================================
# 解释文本
# ============================================================
def build_explanation(payload: dict, result: dict) -> str:
    reasons = []

    if payload["RIDAGEYR"] >= 60:
        reasons.append("your age is in a higher-risk range")
    elif payload["RIDAGEYR"] >= 45:
        reasons.append("your age suggests a moderate baseline cardiovascular risk")

    if payload["BMXBMI"] >= 30:
        reasons.append("your BMI is in the obesity range")
    elif payload["BMXBMI"] >= 25:
        reasons.append("your BMI is above the recommended range")

    if payload["SMQ020"] == 1:
        reasons.append("current smoking increases cardiovascular burden")

    if payload["DIQ010"] == 1:
        reasons.append("a history of diabetes is a major cardiovascular risk factor")

    if payload["BPQ020"] == 1:
        reasons.append("a history of hypertension is strongly associated with cardiovascular risk")

    if payload["PAQ605"] == 2:
        reasons.append("limited moderate physical activity may reduce protective benefit")

    if not reasons:
        reasons.append("your self-reported profile does not show strong major risk flags in this screening model")

    if len(reasons) == 1:
        detail = reasons[0]
    else:
        detail = ", ".join(reasons[:-1]) + ", and " + reasons[-1]

    if result["risk_level"] == "Low":
        lead = "Your screening result is currently in the lower-risk range."
    elif result["risk_level"] == "Moderate":
        lead = "Your screening result suggests a moderate level of cardiovascular risk."
    elif result["risk_level"] == "High":
        lead = "Your screening result indicates an elevated cardiovascular risk profile."
    else:
        lead = "Your screening result places you in a very high-risk group under the current model."

    return f"{lead} This estimate is mainly driven by the following factors: {detail}."


def get_next_step_text(result: dict) -> str:
    level = result["risk_level"]

    if level == "Low":
        return (
            "At this level, the result does not suggest a strong immediate warning signal. "
            "It is still sensible to maintain weight control, regular activity, and periodic blood pressure checks."
        )
    elif level == "Moderate":
        return (
            "At this level, it would be reasonable to review lifestyle habits more closely, "
            "especially smoking status, exercise frequency, and body weight. If you have a family history or symptoms, "
            "a formal medical evaluation may be worthwhile."
        )
    elif level == "High":
        return (
            "At this level, a more formal cardiovascular review is advisable. "
            "You may benefit from checking blood pressure, glucose, and lipid profile if they are not already known."
        )
    else:
        return (
            "At this level, the screening model is flagging a substantially elevated risk profile. "
            "This does not confirm disease, but it does support prompt medical follow-up and objective clinical testing."
        )

# ============================================================
# 图表1：风险概率条形图
# ============================================================
def plot_risk_probability(prob: float):
    fig, ax = plt.subplots(figsize=(6, 1.8))

    ax.barh(["Risk Probability"], [prob], height=0.45)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Estimated Cardiovascular Risk Probability")
    ax.text(min(prob + 0.02, 0.92), 0, f"{prob:.3f}", va="center")

    plt.tight_layout()
    return fig

# ============================================================
# 图表2：风险因子分数图
# 不是模型真实 SHAP，只是前端解释型可视化
# ============================================================
def compute_factor_scores(payload: dict):
    scores = {
        "Age": 0,
        "BMI": 0,
        "Smoking": 0,
        "Diabetes": 0,
        "Hypertension": 0,
        "Physical Inactivity": 0
    }

    age = payload["RIDAGEYR"]
    bmi = payload["BMXBMI"]

    # 年龄
    if age >= 65:
        scores["Age"] = 100
    elif age >= 55:
        scores["Age"] = 75
    elif age >= 45:
        scores["Age"] = 50
    elif age >= 35:
        scores["Age"] = 25

    # BMI
    if bmi >= 35:
        scores["BMI"] = 100
    elif bmi >= 30:
        scores["BMI"] = 75
    elif bmi >= 25:
        scores["BMI"] = 45
    elif bmi >= 23:
        scores["BMI"] = 20

    # 吸烟
    scores["Smoking"] = 100 if payload["SMQ020"] == 1 else 0

    # 糖尿病
    scores["Diabetes"] = 100 if payload["DIQ010"] == 1 else 0

    # 高血压
    scores["Hypertension"] = 100 if payload["BPQ020"] == 1 else 0

    # 缺乏运动
    scores["Physical Inactivity"] = 100 if payload["PAQ605"] == 2 else 0

    return scores


def plot_factor_scores(scores: dict):
    labels = list(scores.keys())
    values = list(scores.values())

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(labels, values)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Relative Risk Contribution Score")
    ax.set_title("Main Risk Factors in Current Screening Profile")

    for i, v in enumerate(values):
        ax.text(min(v + 2, 96), i, str(v), va="center")

    plt.tight_layout()
    return fig

# ============================================================
# 页面主体
# ============================================================
st.markdown('<div class="main-title">Cardiovascular Risk Screening</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A lightweight self-report screening tool for early cardiovascular risk awareness.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
    <b>Purpose</b><br>
    This page estimates whether your profile falls into a higher cardiovascular risk group using a small set of self-reported variables.<br><br>
    <b>Important boundary</b><br>
    It is designed for screening, not diagnosis. A high-risk result does not confirm disease, and a low-risk result does not rule it out.
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Enter your information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=45)
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
    smoking = st.selectbox("Do you currently smoke?", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    diabetes = st.selectbox("Have you been told you have diabetes?", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    hypertension = st.selectbox("Have you been told you have hypertension?", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    activity = st.selectbox("Do you perform moderate physical activity?", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No")

payload = {
    "RIDAGEYR": age,
    "RIAGENDR": sex,
    "BMXBMI": bmi,
    "SMQ020": smoking,
    "DIQ010": diabetes,
    "BPQ020": hypertension,
    "PAQ605": activity
}

if st.button("Generate Risk Assessment"):
    result = predict_user_local(payload)
    explanation = build_explanation(payload, result)
    next_step = get_next_step_text(result)
    risk_class = risk_level_class(result["risk_level"])

    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="margin-top:0;">Screening Result</h3>
            <p><b>Estimated risk probability:</b> {result['risk_probability']:.3f}</p>
            <p><b>Risk category:</b> <span class="{risk_class}">{result['risk_level']}</span></p>
            <p><b>Threshold used by the model:</b> {result['used_threshold']:.3f}</p>
            <p><b>Flagged as higher-risk by model:</b> {result['risk_prediction']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Interpretation")
    st.write(explanation)

    st.subheader("Suggested next step")
    st.write(next_step)

    st.subheader("Chart 1: Probability View")
    fig1 = plot_risk_probability(result["risk_probability"])
    st.pyplot(fig1)

    st.subheader("Chart 2: Main Risk Factor Profile")
    factor_scores = compute_factor_scores(payload)
    fig2 = plot_factor_scores(factor_scores)
    st.pyplot(fig2)

    st.subheader("Important note")
    st.info(
        "This is a screening-oriented estimate. If you have chest pain, shortness of breath, fainting, "
        "or rapidly worsening symptoms, seek medical care directly rather than relying on this tool."
    )

st.markdown(
    '<p class="small-note">This page only keeps the user-input screening model. '
    'The professional / laboratory version has been removed to keep the product experience simple, stable, and easier to interpret.</p>',
    unsafe_allow_html=True
)