import streamlit as st
import pandas as pd
import joblib


# ============================================================
# 页面基础设置
# ============================================================
st.set_page_config(
    page_title="Cardiovascular Risk Screening",
    page_icon="❤️",
    layout="centered"
)


# ============================================================
# 自定义样式：背景、卡片、按钮、提示框
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f8ff 0%, #eef7f1 100%);
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #143d59;
        margin-bottom: 0.3rem;
    }

    .sub-text {
        font-size: 1rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }

    .card {
        background: white;
        padding: 1.2rem 1.2rem 0.8rem 1.2rem;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(20, 61, 89, 0.08);
        margin-bottom: 1rem;
    }

    .result-card {
        background: #ffffff;
        padding: 1.4rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(20, 61, 89, 0.10);
        border-left: 8px solid #2b6cb0;
        margin-top: 1rem;
    }

    .risk-low {
        color: #2f855a;
        font-weight: 700;
    }

    .risk-moderate {
        color: #b7791f;
        font-weight: 700;
    }

    .risk-high {
        color: #dd6b20;
        font-weight: 700;
    }

    .risk-very-high {
        color: #c53030;
        font-weight: 700;
    }

    .small-note {
        color: #718096;
        font-size: 0.92rem;
    }

    div.stButton > button {
        background-color: #2b6cb0;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1.2rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #1f4e85;
        color: white;
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
# 如果模型文件在 models/ 下，请改路径
# ============================================================
@st.cache_resource
def load_user_model():
    model = joblib.load("calibrated_user_model.joblib")
    threshold = joblib.load("calibrated_user_threshold.joblib")["threshold"]
    return model, threshold


# ============================================================
# 本地预测函数
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
# 页面主体
# ============================================================
st.markdown('<div class="main-title">Cardiovascular Risk Screening</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">A lightweight screening tool based on self-reported information. '
    'This page is designed for early risk awareness rather than diagnosis.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card">
    <b>What this tool does</b><br>
    It estimates whether your profile falls into a higher cardiovascular risk group based on a small set of user-reported variables.<br><br>
    <b>What this tool does not do</b><br>
    It does not diagnose heart disease, replace a clinician, or interpret acute symptoms.
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
    risk_class = risk_level_class(result["risk_level"])
    explanation = build_explanation(payload, result)
    next_step = get_next_step_text(result)

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

    st.subheader("Important note")
    st.info(
        "This is a screening-oriented estimate. A high-risk result does not confirm disease, "
        "and a low-risk result does not rule it out. If you have chest pain, shortness of breath, "
        "fainting, or other concerning symptoms, seek medical care directly."
    )

st.markdown(
    '<p class="small-note">Model scope: user-input screening only. '
    'The professional / laboratory version has been removed from this page to keep the experience simple and consistent.</p>',
    unsafe_allow_html=True
)