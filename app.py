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
    layout="wide"
)

# ============================================================
# 样式
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #eef4ff 0%, #f3fbf7 100%);
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #143d59;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #4a5568;
        font-size: 1rem;
        margin-bottom: 1rem;
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

    .advice-box {
        background: white;
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(20, 61, 89, 0.08);
        margin-bottom: 0.8rem;
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
# 多语言字典
# ============================================================
TEXT = {
    "zh": {
        "app_title": "心血管风险筛查",
        "subtitle": "一个基于少量自报信息的轻量化心血管风险早筛工具。",
        "language": "语言",
        "purpose_title": "用途",
        "purpose_text": "本页面使用少量自报变量，估计你的特征是否落入更高的心血管风险人群。",
        "boundary_title": "重要边界",
        "boundary_text": "这是筛查工具，不是诊断工具。高风险结果不代表已经确诊，低风险结果也不代表绝对没有问题。",
        "enter_info": "填写你的信息",
        "age": "年龄",
        "sex": "性别",
        "male": "男",
        "female": "女",
        "bmi": "BMI",
        "smoke": "你目前吸烟吗？",
        "diabetes": "你是否被告知有糖尿病？",
        "hypertension": "你是否被告知有高血压？",
        "activity": "你是否进行中等强度身体活动？",
        "yes": "是",
        "no": "否",
        "button": "生成风险评估",
        "result_title": "筛查结果",
        "probability": "估计风险概率",
        "risk_category": "风险等级",
        "threshold": "模型使用阈值",
        "flagged": "是否被模型标记为较高风险",
        "interpretation": "结果解释",
        "chart1": "图表1：风险概率视图",
        "chart2": "图表2：风险因子雷达对比",
        "chart3": "图表3：与你的数据集基线风险比较",
        "your_profile": "你的画像",
        "reference_profile": "参考低风险画像",
        "advice_title": "个性化建议",
        "important_note": "重要提示",
        "important_note_text": "这是筛查导向的估计。如果你出现胸痛、呼吸困难、晕厥或快速恶化的症状，请直接就医，而不要依赖本工具。",
        "footer": "本页面仅保留用户输入版筛查模型，以保证产品体验更简单、稳定、易解释。",
        "message": "本工具仅用于心血管风险筛查，不构成医疗诊断。"
    },
    "en": {
        "app_title": "Cardiovascular Risk Screening",
        "subtitle": "A lightweight self-report tool for early cardiovascular risk awareness.",
        "language": "Language",
        "purpose_title": "Purpose",
        "purpose_text": "This page uses a small set of self-reported variables to estimate whether your profile falls into a higher cardiovascular risk group.",
        "boundary_title": "Important boundary",
        "boundary_text": "This is a screening tool, not a diagnostic tool. A high-risk result does not confirm disease, and a low-risk result does not rule it out.",
        "enter_info": "Enter your information",
        "age": "Age",
        "sex": "Sex",
        "male": "Male",
        "female": "Female",
        "bmi": "BMI",
        "smoke": "Do you currently smoke?",
        "diabetes": "Have you been told you have diabetes?",
        "hypertension": "Have you been told you have hypertension?",
        "activity": "Do you perform moderate physical activity?",
        "yes": "Yes",
        "no": "No",
        "button": "Generate Risk Assessment",
        "result_title": "Screening Result",
        "probability": "Estimated risk probability",
        "risk_category": "Risk category",
        "threshold": "Threshold used by the model",
        "flagged": "Flagged as higher-risk by model",
        "interpretation": "Interpretation",
        "chart1": "Chart 1: Probability View",
        "chart2": "Chart 2: Radar Comparison of Main Risk Drivers",
        "chart3": "Chart 3: Comparison with Baseline Dataset Risk",
        "your_profile": "Your Profile",
        "reference_profile": "Reference Low-Risk Profile",
        "advice_title": "Personalized Advice",
        "important_note": "Important note",
        "important_note_text": "This is a screening-oriented estimate. If you have chest pain, shortness of breath, fainting, or rapidly worsening symptoms, seek medical care directly rather than relying on this tool.",
        "footer": "This page keeps only the user-input screening model to maintain a simpler and more interpretable product experience.",
        "message": "This tool is for cardiovascular risk screening only and does not constitute a medical diagnosis."
    },
    "es": {
        "app_title": "Detección de Riesgo Cardiovascular",
        "subtitle": "Una herramienta ligera basada en información autodeclarada para la detección temprana del riesgo cardiovascular.",
        "language": "Idioma",
        "purpose_title": "Propósito",
        "purpose_text": "Esta página utiliza un pequeño conjunto de variables autodeclaradas para estimar si tu perfil cae en un grupo de mayor riesgo cardiovascular.",
        "boundary_title": "Límite importante",
        "boundary_text": "Esta es una herramienta de cribado, no de diagnóstico. Un resultado de alto riesgo no confirma enfermedad, y un resultado de bajo riesgo no la descarta.",
        "enter_info": "Introduce tu información",
        "age": "Edad",
        "sex": "Sexo",
        "male": "Hombre",
        "female": "Mujer",
        "bmi": "IMC",
        "smoke": "¿Fumas actualmente?",
        "diabetes": "¿Te han dicho que tienes diabetes?",
        "hypertension": "¿Te han dicho que tienes hipertensión?",
        "activity": "¿Realizas actividad física de intensidad moderada?",
        "yes": "Sí",
        "no": "No",
        "button": "Generar evaluación de riesgo",
        "result_title": "Resultado del cribado",
        "probability": "Probabilidad estimada de riesgo",
        "risk_category": "Categoría de riesgo",
        "threshold": "Umbral utilizado por el modelo",
        "flagged": "Marcado por el modelo como mayor riesgo",
        "interpretation": "Interpretación",
        "chart1": "Gráfico 1: Vista de probabilidad",
        "chart2": "Gráfico 2: Comparación radar de los principales factores de riesgo",
        "chart3": "Gráfico 3: Comparación con el riesgo basal del conjunto de datos",
        "your_profile": "Tu perfil",
        "reference_profile": "Perfil de referencia de bajo riesgo",
        "advice_title": "Consejos personalizados",
        "important_note": "Nota importante",
        "important_note_text": "Esta es una estimación orientada al cribado. Si tienes dolor en el pecho, dificultad para respirar, desmayo o síntomas que empeoran rápidamente, busca atención médica directamente.",
        "footer": "Esta página solo mantiene el modelo de cribado basado en la información del usuario para ofrecer una experiencia más simple y fácil de interpretar.",
        "message": "Esta herramienta es solo para cribado de riesgo cardiovascular y no constituye un diagnóstico médico."
    }
}

# ============================================================
# 语言选择
# ============================================================
lang_map = {
    "中文": "zh",
    "English": "en",
    "Español": "es"
}

top_col1, top_col2 = st.columns([4, 1])
with top_col2:
    lang_label = st.selectbox(TEXT["en"]["language"], options=list(lang_map.keys()))
lang = lang_map[lang_label]
T = TEXT[lang]


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


def translate_risk_level(level: str) -> str:
    mapping = {
        "zh": {
            "Low": "低",
            "Moderate": "中等",
            "High": "高",
            "Very High": "很高"
        },
        "en": {
            "Low": "Low",
            "Moderate": "Moderate",
            "High": "High",
            "Very High": "Very High"
        },
        "es": {
            "Low": "Bajo",
            "Moderate": "Moderado",
            "High": "Alto",
            "Very High": "Muy alto"
        }
    }
    return mapping[lang][level]


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
# 如在 models/ 下请改路径
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
        "message": T["message"]
    }


# ============================================================
# 解释文本
# ============================================================
def build_explanation(payload: dict, result: dict) -> str:
    reasons = []

    if payload["RIDAGEYR"] >= 60:
        reasons.append({
            "zh": "年龄处于更高风险区间",
            "en": "your age is in a higher-risk range",
            "es": "tu edad se encuentra en un rango de mayor riesgo"
        }[lang])
    elif payload["RIDAGEYR"] >= 45:
        reasons.append({
            "zh": "年龄提示存在中等基础心血管风险",
            "en": "your age suggests a moderate baseline cardiovascular risk",
            "es": "tu edad sugiere un riesgo cardiovascular basal moderado"
        }[lang])

    if payload["BMXBMI"] >= 30:
        reasons.append({
            "zh": "BMI处于肥胖范围",
            "en": "your BMI is in the obesity range",
            "es": "tu IMC se encuentra en el rango de obesidad"
        }[lang])
    elif payload["BMXBMI"] >= 25:
        reasons.append({
            "zh": "BMI高于推荐范围",
            "en": "your BMI is above the recommended range",
            "es": "tu IMC está por encima del rango recomendado"
        }[lang])

    if payload["SMQ020"] == 1:
        reasons.append({
            "zh": "当前吸烟会增加心血管负担",
            "en": "current smoking increases cardiovascular burden",
            "es": "el tabaquismo actual aumenta la carga cardiovascular"
        }[lang])

    if payload["DIQ010"] == 1:
        reasons.append({
            "zh": "糖尿病病史是重要心血管风险因素",
            "en": "a history of diabetes is a major cardiovascular risk factor",
            "es": "los antecedentes de diabetes son un factor importante de riesgo cardiovascular"
        }[lang])

    if payload["BPQ020"] == 1:
        reasons.append({
            "zh": "高血压病史与心血管风险密切相关",
            "en": "a history of hypertension is strongly associated with cardiovascular risk",
            "es": "los antecedentes de hipertensión están fuertemente asociados al riesgo cardiovascular"
        }[lang])

    if payload["PAQ605"] == 2:
        reasons.append({
            "zh": "缺乏中等强度身体活动会减少保护作用",
            "en": "limited moderate physical activity may reduce protective benefit",
            "es": "la escasa actividad física moderada puede reducir el efecto protector"
        }[lang])

    if not reasons:
        reasons.append({
            "zh": "当前自报信息中未出现明显的主要风险信号",
            "en": "your self-reported profile does not show strong major risk flags in this screening model",
            "es": "tu perfil autodeclarado no muestra señales fuertes de riesgo principal en este modelo de cribado"
        }[lang])

    if len(reasons) == 1:
        detail = reasons[0]
    else:
        detail = "，".join(reasons[:-1]) + "，以及" + reasons[-1] if lang == "zh" else ", ".join(reasons[:-1]) + ", and " + reasons[-1] if lang == "en" else ", ".join(reasons[:-1]) + " y " + reasons[-1]

    lead_map = {
        "zh": {
            "Low": "你的筛查结果目前处于较低风险范围。",
            "Moderate": "你的筛查结果提示存在中等水平的心血管风险。",
            "High": "你的筛查结果提示心血管风险升高。",
            "Very High": "你的筛查结果在当前模型下属于很高风险组。"
        },
        "en": {
            "Low": "Your screening result is currently in the lower-risk range.",
            "Moderate": "Your screening result suggests a moderate level of cardiovascular risk.",
            "High": "Your screening result indicates an elevated cardiovascular risk profile.",
            "Very High": "Your screening result places you in a very high-risk group under the current model."
        },
        "es": {
            "Low": "Tu resultado de cribado se encuentra actualmente en el rango de menor riesgo.",
            "Moderate": "Tu resultado de cribado sugiere un nivel moderado de riesgo cardiovascular.",
            "High": "Tu resultado de cribado indica un perfil de riesgo cardiovascular elevado.",
            "Very High": "Tu resultado de cribado te sitúa en un grupo de riesgo muy alto según el modelo actual."
        }
    }

    tail_map = {
        "zh": f" 这一估计主要由以下因素驱动：{detail}。",
        "en": f" This estimate is mainly driven by the following factors: {detail}.",
        "es": f" Esta estimación está impulsada principalmente por los siguientes factores: {detail}."
    }

    return lead_map[lang][result["risk_level"]] + tail_map[lang]


# ============================================================
# 个性化建议：至少8类
# ============================================================
def build_segmented_advice(payload: dict, result: dict):
    age = payload["RIDAGEYR"]
    bmi = payload["BMXBMI"]
    smoking = payload["SMQ020"]
    diabetes = payload["DIQ010"]
    htn = payload["BPQ020"]
    activity = payload["PAQ605"]
    level = result["risk_level"]

    adv = []

    def tr(zh, en, es):
        return {"zh": zh, "en": en, "es": es}[lang]

    if age >= 60 and htn == 1:
        adv.append(tr(
            "老年且合并高血压：建议把规律血压监测和长期控制放在优先位置。",
            "Older adult with hypertension: prioritize regular blood pressure monitoring and long-term control.",
            "Adulto mayor con hipertensión: prioriza el control regular de la presión arterial y su manejo a largo plazo."
        ))

    if age >= 60 and activity == 2:
        adv.append(tr(
            "老年且活动不足：建议从可持续的低冲击活动开始，比如步行或固定自行车，而不是偶尔高强度运动。",
            "Older adult with low activity: start with sustainable low-impact exercise such as walking or stationary cycling rather than occasional intense workouts.",
            "Adulto mayor con baja actividad: comienza con ejercicio de bajo impacto y sostenible, como caminar o bicicleta estática, en lugar de sesiones intensas ocasionales."
        ))

    if bmi >= 30 and activity == 2:
        adv.append(tr(
            "肥胖且活动不足：优先目标应是体重控制和规律活动，而不是只做短期突击式改变。",
            "Obesity with insufficient activity: the first priority should be weight control and regular movement rather than short-term intensive changes.",
            "Obesidad con actividad insuficiente: la primera prioridad debe ser el control del peso y el movimiento regular, no cambios intensivos de corto plazo."
        ))

    if smoking == 1:
        adv.append(tr(
            "吸烟人群：戒烟往往是最高收益的干预之一，通常比其他单一生活方式调整更直接。",
            "Current smoker profile: smoking cessation is often one of the highest-yield interventions.",
            "Perfil de fumador actual: dejar de fumar suele ser una de las intervenciones de mayor impacto."
        ))

    if diabetes == 1:
        adv.append(tr(
            "糖尿病相关人群：建议更重视血糖、体重和血压的联动控制。",
            "Diabetes-related profile: place greater emphasis on combined control of glucose, weight, and blood pressure.",
            "Perfil relacionado con diabetes: da mayor importancia al control combinado de glucosa, peso y presión arterial."
        ))

    if htn == 1 and bmi >= 25:
        adv.append(tr(
            "高血压合并超重：建议同时管理体重和血压，而不是只盯住单一指标。",
            "Hypertension plus excess weight: manage both weight and blood pressure rather than focusing on only one indicator.",
            "Hipertensión con exceso de peso: maneja tanto el peso como la presión arterial en lugar de centrarte en un solo indicador."
        ))

    if age < 45 and smoking == 1:
        adv.append(tr(
            "年轻吸烟人群：虽然年龄仍有一定保护作用，但吸烟会提前推高未来风险。",
            "Younger smoker profile: age still offers some protection, but smoking can push future risk upward earlier than expected.",
            "Perfil de fumador joven: la edad todavía ofrece cierta protección, pero fumar puede aumentar el riesgo futuro antes de lo esperado."
        ))

    if age < 45 and bmi >= 30:
        adv.append(tr(
            "年轻肥胖人群：当前概率未必极高，但这是风险长期积累最值得干预的阶段。",
            "Younger obesity-dominant profile: the immediate probability may not be extreme, but this is an important stage for early intervention.",
            "Perfil joven con obesidad predominante: la probabilidad inmediata puede no ser extrema, pero esta es una etapa importante para intervenir pronto."
        ))

    if diabetes == 1 and htn == 1:
        adv.append(tr(
            "糖尿病合并高血压：这是临床上更值得重视的风险组合，建议更系统地随访。",
            "Diabetes plus hypertension: this is a clinically important risk cluster and supports more structured follow-up.",
            "Diabetes más hipertensión: este es un grupo de riesgo clínicamente importante y justifica un seguimiento más estructurado."
        ))

    if activity == 1 and smoking == 2 and bmi < 25 and level in ["Low", "Moderate"]:
        adv.append(tr(
            "保护性生活方式画像：当前模式有一些积极信号，重点在于持续保持，而不是激进调整。",
            "Protective lifestyle profile: your current pattern shows favorable signals, so the priority is maintenance rather than aggressive intervention.",
            "Perfil de estilo de vida protector: tu patrón actual muestra señales favorables, por lo que la prioridad es mantenerlo."
        ))

    if level == "Very High":
        adv.append(tr(
            "很高风险组：虽然本工具不是诊断工具，但这个结果支持尽快进行正式临床评估和客观检测。",
            "Very high screening category: although this tool is not diagnostic, this result supports prompt clinical evaluation and objective testing.",
            "Categoría de riesgo muy alto: aunque esta herramienta no es diagnóstica, este resultado respalda una evaluación clínica rápida y pruebas objetivas."
        ))

    if level == "High":
        adv.append(tr(
            "高风险组：如果尚未了解自己的血压、血糖和血脂情况，建议尽快补充这些客观信息。",
            "High screening category: if you do not already know your blood pressure, glucose, and lipid status, it would be reasonable to obtain them.",
            "Categoría de alto riesgo: si aún no conoces tu presión arterial, glucosa y perfil lipídico, sería razonable obtenerlos."
        ))

    if not adv:
        adv.append(tr(
            "当前没有触发特别明确的分层特征，建议继续保持体重管理、规律活动和基础体检。",
            "No strongly specific subgroup pattern was triggered. Continue weight control, regular activity, and routine health checks.",
            "No se activó un patrón de subgrupo muy específico. Continúa con el control del peso, la actividad regular y los chequeos rutinarios."
        ))

    return adv[:8]


# ============================================================
# 图表1：风险概率条形图
# ============================================================
def plot_risk_probability(prob: float):
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.barh([T["probability"]], [prob], height=0.45)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(T["chart1"])
    ax.text(min(prob + 0.02, 0.92), 0, f"{prob:.3f}", va="center")
    plt.tight_layout()
    return fig


# ============================================================
# 雷达图数据
# ============================================================
def compute_factor_scores(payload: dict):
    scores = {
        "Age": 0,
        "BMI": 0,
        "Smoking": 0,
        "Diabetes": 0,
        "Hypertension": 0,
        "Inactivity": 0
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
    scores["Inactivity"] = 100 if payload["PAQ605"] == 2 else 0

    return scores


def compute_reference_scores(payload: dict):
    scores = {
        "Age": 0,
        "BMI": 15,
        "Smoking": 0,
        "Diabetes": 0,
        "Hypertension": 0,
        "Inactivity": 10
    }

    age = payload["RIDAGEYR"]

    if age >= 65:
        scores["Age"] = 100
    elif age >= 55:
        scores["Age"] = 75
    elif age >= 45:
        scores["Age"] = 50
    elif age >= 35:
        scores["Age"] = 25
    else:
        scores["Age"] = 10

    return scores


def plot_small_radar_chart(scores: dict, title: str):
    labels = list(scores.keys())
    values = list(scores.values())

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.22)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title(title, y=1.10, fontsize=11)

    plt.tight_layout()
    return fig


# ============================================================
# 图表3：与基线风险比较
# ============================================================
def plot_baseline_comparison(prob: float, baseline=0.045):
    labels = [
        {"zh": "你的估计风险", "en": "Estimated Risk", "es": "Riesgo estimado"}[lang],
        {"zh": "数据集基线风险", "en": "Dataset Baseline Risk", "es": "Riesgo basal del conjunto"}[lang]
    ]
    values = [prob, baseline]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(labels, values)
    ax.set_ylim(0, max(0.2, prob + 0.05))
    ax.set_ylabel("Probability")
    ax.set_title(T["chart3"])

    for i, v in enumerate(values):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")

    plt.tight_layout()
    return fig


# ============================================================
# 页面头部
# ============================================================
st.markdown(f'<div class="main-title">{T["app_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{T["subtitle"]}</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="info-card">
    <b>{T["purpose_title"]}</b><br>
    {T["purpose_text"]}<br><br>
    <b>{T["boundary_title"]}</b><br>
    {T["boundary_text"]}
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 输入区域
# ============================================================
st.subheader(T["enter_info"])

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(T["age"], min_value=20, max_value=100, value=45)
    sex = st.selectbox(T["sex"], options=[1, 2], format_func=lambda x: T["male"] if x == 1 else T["female"])
    bmi = st.number_input(T["bmi"], min_value=10.0, max_value=60.0, value=24.0)
    smoking = st.selectbox(T["smoke"], options=[1, 2], format_func=lambda x: T["yes"] if x == 1 else T["no"])

with col2:
    diabetes = st.selectbox(T["diabetes"], options=[1, 2], format_func=lambda x: T["yes"] if x == 1 else T["no"])
    hypertension = st.selectbox(T["hypertension"], options=[1, 2], format_func=lambda x: T["yes"] if x == 1 else T["no"])
    activity = st.selectbox(T["activity"], options=[1, 2], format_func=lambda x: T["yes"] if x == 1 else T["no"])

payload = {
    "RIDAGEYR": age,
    "RIAGENDR": sex,
    "BMXBMI": bmi,
    "SMQ020": smoking,
    "DIQ010": diabetes,
    "BPQ020": hypertension,
    "PAQ605": activity
}

# ============================================================
# 输出区域
# ============================================================
if st.button(T["button"]):
    result = predict_user_local(payload)
    risk_class = risk_level_class(result["risk_level"])
    risk_level_translated = translate_risk_level(result["risk_level"])
    explanation = build_explanation(payload, result)
    advice_list = build_segmented_advice(payload, result)

    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="margin-top:0;">{T["result_title"]}</h3>
            <p><b>{T["probability"]}:</b> {result['risk_probability']:.3f}</p>
            <p><b>{T["risk_category"]}:</b> <span class="{risk_class}">{risk_level_translated}</span></p>
            <p><b>{T["threshold"]}:</b> {result['used_threshold']:.3f}</p>
            <p><b>{T["flagged"]}:</b> {result['risk_prediction']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(T["interpretation"])
    st.write(explanation)

    # 图表1
    st.subheader(T["chart1"])
    fig1 = plot_risk_probability(result["risk_probability"])
    st.pyplot(fig1)

    # 图表2：两个小雷达图并排
    st.subheader(T["chart2"])
    user_scores = compute_factor_scores(payload)
    ref_scores = compute_reference_scores(payload)

    radar_col1, radar_col2 = st.columns(2)
    with radar_col1:
        fig_user = plot_small_radar_chart(user_scores, T["your_profile"])
        st.pyplot(fig_user)

    with radar_col2:
        fig_ref = plot_small_radar_chart(ref_scores, T["reference_profile"])
        st.pyplot(fig_ref)

    # 图表3
    st.subheader(T["chart3"])
    fig3 = plot_baseline_comparison(result["risk_probability"], baseline=0.045)
    st.pyplot(fig3)

    # 建议
    st.subheader(T["advice_title"])
    for i, advice in enumerate(advice_list, start=1):
        st.markdown(
            f"""
            <div class="advice-box">
                <b>{i}.</b> {advice}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader(T["important_note"])
    st.info(T["important_note_text"])

st.markdown(
    f'<p class="small-note">{T["footer"]}</p>',
    unsafe_allow_html=True
)