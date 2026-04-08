import streamlit as st
import pandas as pd
import joblib

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


# ============================================================
# 本地模型加载
# 正式网站上线后，这块可以替换成 API 请求
# ============================================================
@st.cache_resource
def load_user_model():
    model = joblib.load("calibrated_user_model.joblib")
    threshold = joblib.load("calibrated_user_threshold.joblib")["threshold"]
    return model, threshold


@st.cache_resource
def load_full_model():
    model = joblib.load("calibrated_full_model.joblib")
    preprocessor = joblib.load("calibrated_full_preprocessor.joblib")
    threshold = joblib.load("calibrated_full_threshold.joblib")["threshold"]
    return model, preprocessor, threshold


# ============================================================
# 预留网站接口主体
# 后面正式接网站时，把这两个函数替换成 requests.post(...)
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


def predict_full_local(payload: dict):
    model, preprocessor, threshold = load_full_model()
    sample_df = pd.DataFrame([payload])

    X_t = preprocessor.transform(sample_df)
    prob = model.predict_proba(X_t)[:, 1][0]
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
# 页面
# ============================================================
st.set_page_config(page_title="Cardiovascular Risk Screening", layout="centered")

st.title("Cardiovascular Risk Screening")
st.caption("Prototype version")

st.warning("This tool is for risk screening only and does not constitute a medical diagnosis.")

mode = st.radio(
    "Choose prediction mode",
    ["User Input Model", "Full Variable Model"]
)

# ============================================================
# 用户模型页面
# ============================================================
if mode == "User Input Model":
    st.subheader("Basic Screening")

    age = st.number_input("Age", min_value=20, max_value=100, value=45)
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)

    smoking = st.selectbox("Do you currently smoke?", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    diabetes = st.selectbox("Have you been told you have diabetes?", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    hypertension = st.selectbox("Have you been told you have hypertension?", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    activity = st.selectbox("Do you perform moderate physical activity?", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button("Predict with User Model"):
        payload = {
            "RIDAGEYR": age,
            "RIAGENDR": sex,
            "BMXBMI": bmi,
            "SMQ020": smoking,
            "DIQ010": diabetes,
            "BPQ020": hypertension,
            "PAQ605": activity
        }

        result = predict_user_local(payload)

        st.subheader("Result")
        st.write(f"Risk probability: **{result['risk_probability']:.3f}**")
        st.write(f"Risk level: **{result['risk_level']}**")
        st.write(f"High-risk flag: **{result['risk_prediction']}**")
        st.write(f"Threshold used: **{result['used_threshold']:.3f}**")

        if result["risk_prediction"] == 1:
            st.error("You are flagged as higher risk under the current screening threshold.")
        else:
            st.success("You are not flagged by the current screening threshold.")

        st.caption(result["message"])


# ============================================================
# 完整模型页面
# ============================================================
else:
    st.subheader("Extended Screening with Clinical Variables")

    age = st.number_input("Age", min_value=20, max_value=100, value=55, key="full_age")
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female", key="full_sex")
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, key="full_bmi")
    sbp = st.number_input("Systolic Blood Pressure (SBP)", min_value=70.0, max_value=250.0, value=130.0)

    tchol = st.number_input("Total Cholesterol", min_value=50.0, max_value=500.0, value=180.0)
    trigly = st.number_input("Triglycerides", min_value=20.0, max_value=1000.0, value=120.0)
    hdl = st.number_input("HDL Cholesterol", min_value=10.0, max_value=150.0, value=50.0)
    glu = st.number_input("Plasma Glucose", min_value=40.0, max_value=500.0, value=95.0)
    hba1c = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=5.5)

    if st.button("Predict with Full Model"):
        payload = {
            "RIDAGEYR": age,
            "RIAGENDR": sex,
            "BMXBMI": bmi,
            "SBP": sbp,
            "LBXTC": tchol,
            "LBXTR": trigly,
            "LBDHDD": hdl,
            "LBXGLU": glu,
            "LBXGH": hba1c
        }

        result = predict_full_local(payload)

        st.subheader("Result")
        st.write(f"Risk probability: **{result['risk_probability']:.3f}**")
        st.write(f"Risk level: **{result['risk_level']}**")
        st.write(f"High-risk flag: **{result['risk_prediction']}**")
        st.write(f"Threshold used: **{result['used_threshold']:.3f}**")

        if result["risk_prediction"] == 1:
            st.error("You are flagged as higher risk under the current screening threshold.")
        else:
            st.success("You are not flagged by the current screening threshold.")

        st.caption(result["message"])