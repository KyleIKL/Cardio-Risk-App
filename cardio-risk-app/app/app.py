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
# 页面样式
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #eef4ff 0%, #f3fbf7 100%);
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #143d59;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #4a5568;
        font-size: 1rem;
        margin-bottom: 1.0rem;
    }

    .info-card {
        background: white;
        padding: 1.2rem 1.2rem 0.9rem 1.2rem;
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

    .small-note {
        color: #718096;
        font-size: 0.92rem;
    }

    .advice-box {
        background: #ffffff;
        border-radius: 16px;
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: 0 6px 18px rgba(20, 61, 89, 0.08);
        margin-bottom: 1rem;
    }

    div.stButton > button {
        background-color: #2b6cb0;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.72rem 1.2rem;
        font-weight: 700;
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
# 文本解释
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


# ============================================================
# 至少 8 套差异化建议
# ============================================================
def build_segmented_advice(payload: dict, result: dict):
    age = payload["RIDAGEYR"]
    bmi = payload["BMXBMI"]
    smoking = payload["SMQ020"]
    diabetes = payload["DIQ010"]
    htn = payload["BPQ020"]
    activity = payload["PAQ605"]
    level = result["risk_level"]

    advice_list = []

    # 1
    if age >= 60 and htn == 1:
        advice_list.append(
            "Older adult with hypertension: prioritize regular blood pressure monitoring, medication adherence if already prescribed, and a clinician-reviewed blood pressure target."
        )

    # 2
    if age >= 60 and activity == 2:
        advice_list.append(
            "Older adult with low activity: introduce low-impact but consistent activity, such as walking, stationary cycling, or supervised exercise, rather than relying on occasional intense exercise."
        )

    # 3
    if bmi >= 30 and activity == 2:
        advice_list.append(
            "Obesity with insufficient activity: the first target should be sustainable weight reduction through calorie control and routine movement, because this combination often drives long-term cardiovascular burden."
        )

    # 4
    if smoking == 1:
        advice_list.append(
            "Current smoker profile: smoking cessation is likely one of the highest-yield interventions in your case. Even before other optimization, reducing smoking exposure can materially lower future cardiovascular burden."
        )

    # 5
    if diabetes == 1:
        advice_list.append(
            "Diabetes-related profile: this result supports closer control of glucose status, weight, and blood pressure, because cardiovascular risk in diabetes is often driven by multiple interacting factors."
        )

    # 6
    if htn == 1 and bmi >= 25:
        advice_list.append(
            "Hypertension plus excess weight: this combination is a strong argument for both blood pressure management and gradual weight reduction rather than focusing on a single lifestyle change."
        )

    # 7
    if age < 45 and smoking == 1:
        advice_list.append(
            "Younger smoker profile: even if age still provides some protection, smoking can shift future risk upward earlier than expected, so early behavior change has a high long-run payoff."
        )

    # 8
    if age < 45 and bmi >= 30:
        advice_list.append(
            "Younger obesity-dominant profile: your immediate probability may still be limited by age, but this is exactly the stage where weight-related risk can accumulate silently over time."
        )

    # 9
    if diabetes == 1 and htn == 1:
        advice_list.append(
            "Diabetes plus hypertension profile: this is a clinically important risk cluster, and it usually justifies more structured follow-up rather than lifestyle changes alone."
        )

    # 10
    if activity == 1 and smoking == 2 and bmi < 25 and level in ["Low", "Moderate"]:
        advice_list.append(
            "Protective lifestyle profile: your current pattern shows some favorable signals. The priority here is maintenance rather than aggressive intervention."
        )

    # 11
    if level == "Very High":
        advice_list.append(
            "Very high screening category: this tool is not diagnostic, but at this level it is reasonable to seek formal clinical assessment and objective testing rather than relying on self-monitoring alone."
        )

    # 12
    if level == "High":
        advice_list.append(
            "High screening category: it would be reasonable to review blood pressure, glucose, and lipid status if those are not already known."
        )

    if not advice_list:
        advice_list.append(
            "No strongly specific subgroup pattern was triggered. A general priority would be maintaining healthy body weight, regular activity, and periodic clinical screening."
        )

    return advice_list[:8]  # 最终最多展示 8 条


# ============================================================
# 图表1：风险概率条形图
# ============================================================
def plot_risk_probability(prob: float):
    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    ax.barh(["Risk Probability"], [prob], height=0.45)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Estimated Cardiovascular Risk Probability")
    ax.text(min(prob + 0.02, 0.92), 0, f"{prob:.3f}", va="center")
    plt.tight_layout()
    return fig


# ============================================================
# 图表2：雷达图
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

    if age >= 65:
        scores["Age"] = 100
    elif age >= 55:
        scores["Age"] = 75
    elif age >= 45:
        scores["Age"] = 50
    elif age >= 35:
        scores["Age"] = 25

    if bmi >= 35:
        scores["BMI"] = 100
    elif bmi >= 30:
        scores["BMI"] = 75
    elif bmi >= 25:
        scores["BMI"] = 45
    elif bmi >= 23:
        scores["BMI"] = 20

    scores["Smoking"] = 100 if payload["SMQ020"] == 1 else 0
    scores["Diabetes"] = 100 if payload["DIQ010"] == 1 else 0
    scores["Hypertension"] = 100 if payload["BPQ020"] == 1 else 0
    scores["Physical Inactivity"] = 100 if payload["PAQ605"] == 2 else 0

    return scores


def plot_radar_chart(scores: dict):
    labels = list(scores.keys())
    values = list(scores.values())

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.8, 5.8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_title("Risk Factor Radar Profile", y=1.08)

    plt.tight_layout()
    return fig


# ============================================================
# 图表3：与基线患病率比较
# ============================================================
def plot_baseline_comparison(prob: float, baseline=0.045):
    labels = ["Estimated Risk", "Dataset Baseline Risk"]
    values = [prob, baseline]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(labels, values)
    ax.set_ylim(0, max(0.2, prob + 0.05))
    ax.set_ylabel("Probability")
    ax.set_title("Your Estimated Risk vs Baseline Dataset Risk")

    for i, v in enumerate(values):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")

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
    risk_class = risk_level_class(result["risk_level"])
    explanation = build_explanation(payload, result)
    advice_list = build_segmented_advice(payload, result)

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

    st.subheader("Chart 1: Probability View")
    fig1 = plot_risk_probability(result["risk_probability"])
    st.pyplot(fig1)

    st.subheader("Chart 2: Radar Profile of Main Risk Drivers")
    scores = compute_factor_scores(payload)
    fig2 = plot_radar_chart(scores)
    st.pyplot(fig2)

    st.subheader("Chart 3: Comparison with Baseline Risk")
    fig3 = plot_baseline_comparison(result["risk_probability"], baseline=0.045)
    st.pyplot(fig3)

    st.subheader("Personalized Advice")
    for i, advice in enumerate(advice_list, start=1):
        st.markdown(
            f"""
            <div class="advice-box">
                <b>Recommendation {i}</b><br>
                {advice}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Important note")
    st.info(
        "This is a screening-oriented estimate. If you have chest pain, shortness of breath, fainting, "
        "or rapidly worsening symptoms, seek medical care directly rather than relying on this tool."
    )

st.markdown(
    '<p class="small-note">This page keeps only the user-input screening model to maintain a simpler and more interpretable product experience.</p>',
    unsafe_allow_html=True
)