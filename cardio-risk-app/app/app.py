import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# ============================================================
# Matplotlib 设置
# 不再依赖中文字体在图里直接渲染
# 图内文本统一用英文短标签，避免中文方框问题
# ============================================================
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial",
    "Liberation Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False

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

    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.95rem;
        margin-left: 0.4rem;
    }

    .badge-low {
        background: #d7f5df;
        color: #1f7a3d;
    }

    .badge-moderate {
        background: #fff3cd;
        color: #9a6700;
    }

    .badge-high {
        background: #ffe2c7;
        color: #b45309;
    }

    .badge-very-high {
        background: #ffd6d6;
        color: #b91c1c;
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
# 多语言文本
# ============================================================
TEXT = {
    "zh": {
        "app_title": "心血管风险筛查",
        "subtitle": "一个基于少量自报信息的轻量化心血管风险早筛工具。",
        "purpose_title": "用途",
        "purpose_text": "本页面使用少量自报变量，估计你的特征是否落入更高的心血管风险人群。这个结果更适合用作风险预警和健康管理参考，而不是医疗诊断结论。",
        "boundary_title": "重要边界",
        "boundary_text": "这是筛查工具，不是诊断工具。高风险结果不代表已经确诊，低风险结果也不代表绝对没有问题。如果你已经有胸痛、明显气短、晕厥、心悸加重或活动后不适等症状，应该优先就医。",
        "enter_info": "填写你的信息",
        "age": "年龄",
        "sex": "性别",
        "male": "男",
        "female": "女",
        "height": "身高（cm）",
        "weight": "体重（kg）",
        "bmi_auto": "自动计算 BMI",
        "smoke": "你目前吸烟吗？",
        "diabetes": "你是否被告知有糖尿病？",
        "hypertension": "你是否被告知有高血压？",
        "activity": "你是否进行中等强度身体活动？",
        "family_history": "是否有心血管家族史？",
        "alcohol": "你是否经常饮酒？",
        "yes": "是",
        "no": "否",
        "none": "无",
        "occasional": "偶尔",
        "frequent": "经常",
        "button": "生成风险评估",
        "result_title": "筛查结果",
        "probability": "估计风险概率",
        "risk_category": "风险等级",
        "threshold": "模型使用阈值",
        "flagged": "是否被模型标记为较高风险",
        "interpretation": "结果解释",
        "how_to_read": "如何理解这个结果",
        "chart1_title": "图表1：风险概率视图",
        "chart2_title": "图表2：你的风险因子画像 vs 参考低风险画像",
        "chart3_title": "图表3：你的估计风险 vs 数据集基线风险",
        "your_profile": "你的画像",
        "reference_profile": "参考低风险画像",
        "advice_title": "个性化建议",
        "important_note": "重要提示",
        "important_note_text": "这是筛查导向的估计，不是医疗诊断。如果你存在胸痛、呼吸困难、晕厥、持续胸闷或症状快速加重，请直接就医，而不要依赖本工具。",
        "footer": "本页面仅保留用户输入版筛查模型，以保证产品体验更简单、稳定、易解释。新增的家族史与饮酒选项当前用于解释和建议，不进入模型打分。",
        "message": "本工具仅用于心血管风险筛查，不构成医疗诊断。"
    },
    "en": {
        "app_title": "Cardiovascular Risk Screening",
        "subtitle": "A lightweight self-report tool for early cardiovascular risk awareness.",
        "purpose_title": "Purpose",
        "purpose_text": "This page uses a small set of self-reported variables to estimate whether your profile falls into a higher cardiovascular risk group. The result should be interpreted as an early-warning and health-management reference rather than a medical diagnosis.",
        "boundary_title": "Important boundary",
        "boundary_text": "This is a screening tool, not a diagnostic tool. A high-risk result does not confirm disease, and a low-risk result does not rule it out. If you already have chest pain, marked shortness of breath, fainting, worsening palpitations, or exertional discomfort, you should seek medical care directly.",
        "enter_info": "Enter your information",
        "age": "Age",
        "sex": "Sex",
        "male": "Male",
        "female": "Female",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "bmi_auto": "Calculated BMI",
        "smoke": "Do you currently smoke?",
        "diabetes": "Have you been told you have diabetes?",
        "hypertension": "Have you been told you have hypertension?",
        "activity": "Do you perform moderate physical activity?",
        "family_history": "Family history of cardiovascular disease?",
        "alcohol": "Do you drink alcohol frequently?",
        "yes": "Yes",
        "no": "No",
        "none": "None",
        "occasional": "Occasional",
        "frequent": "Frequent",
        "button": "Generate Risk Assessment",
        "result_title": "Screening Result",
        "probability": "Estimated risk probability",
        "risk_category": "Risk category",
        "threshold": "Threshold used by the model",
        "flagged": "Flagged as higher-risk by model",
        "interpretation": "Interpretation",
        "how_to_read": "How to read this result",
        "chart1_title": "Chart 1: Probability View",
        "chart2_title": "Chart 2: Your Risk Profile vs Reference Low-Risk Profile",
        "chart3_title": "Chart 3: Your Estimated Risk vs Baseline Dataset Risk",
        "your_profile": "Your Profile",
        "reference_profile": "Reference Low-Risk Profile",
        "advice_title": "Personalized Advice",
        "important_note": "Important note",
        "important_note_text": "This is a screening-oriented estimate, not a medical diagnosis. If you have chest pain, shortness of breath, fainting, persistent chest discomfort, or rapidly worsening symptoms, seek medical care directly.",
        "footer": "This page keeps only the user-input screening model to maintain a simpler and more interpretable product experience. The added family-history and alcohol fields are currently used for interpretation and advice, not model scoring.",
        "message": "This tool is for cardiovascular risk screening only and does not constitute a medical diagnosis."
    },
    "es": {
        "app_title": "Detección de Riesgo Cardiovascular",
        "subtitle": "Una herramienta ligera basada en información autodeclarada para la detección temprana del riesgo cardiovascular.",
        "purpose_title": "Propósito",
        "purpose_text": "Esta página utiliza un pequeño conjunto de variables autodeclaradas para estimar si tu perfil cae en un grupo de mayor riesgo cardiovascular. El resultado debe interpretarse como una referencia de alerta temprana y de gestión de la salud, no como un diagnóstico médico.",
        "boundary_title": "Límite importante",
        "boundary_text": "Esta es una herramienta de cribado, no de diagnóstico. Un resultado de alto riesgo no confirma enfermedad, y un resultado de bajo riesgo no la descarta. Si ya presentas dolor torácico, dificultad respiratoria marcada, desmayo, palpitaciones en empeoramiento o malestar con el esfuerzo, debes buscar atención médica directa.",
        "enter_info": "Introduce tu información",
        "age": "Edad",
        "sex": "Sexo",
        "male": "Hombre",
        "female": "Mujer",
        "height": "Altura (cm)",
        "weight": "Peso (kg)",
        "bmi_auto": "IMC calculado",
        "smoke": "¿Fumas actualmente?",
        "diabetes": "¿Te han dicho que tienes diabetes?",
        "hypertension": "¿Te han dicho que tienes hipertensión?",
        "activity": "¿Realizas actividad física moderada?",
        "family_history": "¿Antecedentes familiares de enfermedad cardiovascular?",
        "alcohol": "¿Consumes alcohol con frecuencia?",
        "yes": "Sí",
        "no": "No",
        "none": "Ninguno",
        "occasional": "Ocasional",
        "frequent": "Frecuente",
        "button": "Generar evaluación de riesgo",
        "result_title": "Resultado del cribado",
        "probability": "Probabilidad estimada de riesgo",
        "risk_category": "Categoría de riesgo",
        "threshold": "Umbral utilizado por el modelo",
        "flagged": "Marcado por el modelo como mayor riesgo",
        "interpretation": "Interpretación",
        "how_to_read": "Cómo interpretar este resultado",
        "chart1_title": "Gráfico 1: Vista de probabilidad",
        "chart2_title": "Gráfico 2: Tu perfil de riesgo vs perfil de referencia de bajo riesgo",
        "chart3_title": "Gráfico 3: Tu riesgo estimado vs riesgo basal del conjunto de datos",
        "your_profile": "Tu perfil",
        "reference_profile": "Perfil de referencia de bajo riesgo",
        "advice_title": "Consejos personalizados",
        "important_note": "Nota importante",
        "important_note_text": "Esta es una estimación orientada al cribado, no un diagnóstico médico. Si tienes dolor en el pecho, dificultad respiratoria, desmayo, molestias torácicas persistentes o síntomas que empeoran rápidamente, busca atención médica directa.",
        "footer": "Esta página solo mantiene el modelo de cribado basado en la información del usuario para ofrecer una experiencia más simple y fácil de interpretar. Los campos añadidos de antecedentes familiares y alcohol se usan actualmente para la interpretación y los consejos, no para la puntuación del modelo.",
        "message": "Esta herramienta es solo para cribado de riesgo cardiovascular y no constituye un diagnóstico médico."
    }
}

lang_map = {"中文": "zh", "English": "en", "Español": "es"}
top_col1, top_col2 = st.columns([4, 1])
with top_col2:
    lang_label = st.selectbox("Language", options=list(lang_map.keys()))
lang = lang_map[lang_label]
T = TEXT[lang]

# ============================================================
# 风险分层与徽章
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
        "zh": {"Low": "低", "Moderate": "中等", "High": "高", "Very High": "很高"},
        "en": {"Low": "Low", "Moderate": "Moderate", "High": "High", "Very High": "Very High"},
        "es": {"Low": "Bajo", "Moderate": "Moderado", "High": "Alto", "Very High": "Muy alto"}
    }
    return mapping[lang][level]

def risk_badge_class(level: str) -> str:
    mapping = {
        "Low": "badge-low",
        "Moderate": "badge-moderate",
        "High": "badge-high",
        "Very High": "badge-very-high"
    }
    return mapping[level]

# ============================================================
# 模型加载
# 如在 models/ 下，请改路径
# ============================================================
@st.cache_resource
def load_user_model():
    model = joblib.load("calibrated_user_model.joblib")
    threshold = joblib.load("calibrated_user_threshold.joblib")["threshold"]
    return model, threshold

# ============================================================
# 预测
# ============================================================
def predict_user_local(payload_model: dict):
    model, threshold = load_user_model()
    sample_df = pd.DataFrame([payload_model])

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
# 解释文本：明显加长
# ============================================================
def build_explanation(payload_all: dict, result: dict) -> str:
    def tr(zh, en, es):
        return {"zh": zh, "en": en, "es": es}[lang]

    drivers = []

    if payload_all["RIDAGEYR"] >= 60:
        drivers.append(tr(
            "年龄已经进入典型的心血管风险上升区间",
            "age has already entered a typical range where cardiovascular risk rises more noticeably",
            "la edad ya se encuentra en un intervalo típico donde el riesgo cardiovascular aumenta de forma más evidente"
        ))
    elif payload_all["RIDAGEYR"] >= 45:
        drivers.append(tr(
            "年龄提示基础风险已不再属于最低水平",
            "age suggests that baseline risk is no longer in the lowest range",
            "la edad sugiere que el riesgo basal ya no se encuentra en el nivel más bajo"
        ))

    if payload_all["BMXBMI"] >= 30:
        drivers.append(tr(
            "体重状态已经进入肥胖范围，这通常会和代谢、血压及长期心血管负担一起变化",
            "weight status is already in the obesity range, which often moves together with metabolism, blood pressure, and long-term cardiovascular burden",
            "el estado de peso ya se encuentra en el rango de obesidad, lo que suele ir acompañado de cambios en el metabolismo, la presión arterial y la carga cardiovascular a largo plazo"
        ))
    elif payload_all["BMXBMI"] >= 25:
        drivers.append(tr(
            "BMI高于理想范围，提示体重管理仍有改善空间",
            "BMI is above the ideal range, suggesting there is still room for improvement in weight management",
            "el IMC está por encima del rango ideal, lo que sugiere que todavía hay margen de mejora en el control del peso"
        ))

    if payload_all["SMQ020"] == 1:
        drivers.append(tr(
            "吸烟会持续增加心血管系统负担，并放大其他风险因素的影响",
            "smoking adds ongoing burden to the cardiovascular system and can amplify the impact of other risk factors",
            "fumar añade una carga continua al sistema cardiovascular y puede amplificar el impacto de otros factores de riesgo"
        ))

    if payload_all["DIQ010"] == 1:
        drivers.append(tr(
            "糖尿病病史是心血管风险评估中非常重要的结构性因素之一",
            "a history of diabetes is one of the more structurally important factors in cardiovascular risk assessment",
            "los antecedentes de diabetes son uno de los factores estructuralmente más importantes en la evaluación del riesgo cardiovascular"
        ))

    if payload_all["BPQ020"] == 1:
        drivers.append(tr(
            "高血压病史通常意味着血管长期承受更高压力，这在风险形成中很关键",
            "a history of hypertension often means the vascular system has been exposed to higher long-term pressure, which is highly relevant for risk formation",
            "los antecedentes de hipertensión suelen implicar que el sistema vascular ha estado expuesto a una presión más alta a largo plazo, lo que es muy relevante para la formación del riesgo"
        ))

    if payload_all["PAQ605"] == 2:
        drivers.append(tr(
            "中等强度活动不足意味着保护性因素偏弱",
            "insufficient moderate physical activity suggests that protective lifestyle factors are weaker than ideal",
            "la falta de actividad física moderada sugiere que los factores protectores del estilo de vida son más débiles de lo ideal"
        ))

    if payload_all["FAMILY_HISTORY"] == 1:
        drivers.append(tr(
            "家族史提示存在额外背景风险，即使它没有直接进入当前模型打分，也应在解释上被认真考虑",
            "family history suggests additional background risk, and even though it is not directly scored by the current model, it should still be taken seriously in interpretation",
            "los antecedentes familiares sugieren un riesgo de fondo adicional y, aunque no entren directamente en la puntuación del modelo actual, deben considerarse seriamente en la interpretación"
        ))

    if payload_all["ALCOHOL_LEVEL"] == 3:
        drivers.append(tr(
            "经常饮酒可能进一步增加整体风险负担，尤其是在已有其他风险因素时",
            "frequent alcohol intake may further increase overall risk burden, especially when other risk factors are already present",
            "el consumo frecuente de alcohol puede aumentar aún más la carga global de riesgo, especialmente cuando ya existen otros factores de riesgo"
        ))

    if not drivers:
        drivers.append(tr(
            "当前自报信息中没有出现非常强的主要风险信号",
            "the current self-reported profile does not show very strong major risk signals",
            "el perfil autodeclarado actual no muestra señales de riesgo principal especialmente fuertes"
        ))

    if lang == "zh":
        detail = "；".join(drivers)
        intro = {
            "Low": "当前结果提示你的风险大体仍处于较低区间。",
            "Moderate": "当前结果提示你的风险已经高于较低基线水平，但还不一定意味着立即存在明确疾病。",
            "High": "当前结果说明你的风险特征已经出现比较明显的累积，需要比普通人更认真地看待后续管理。",
            "Very High": "当前结果说明在这套筛查模型下，你落入了显著偏高的风险分层。"
        }[result["risk_level"]]

        meaning = {
            "Low": "这通常意味着从自报变量角度看，强风险信号不多，但这并不等于绝对安全。",
            "Moderate": "这通常意味着部分危险因素已经开始叠加，风险管理的收益会明显增加。",
            "High": "这通常意味着你不只是存在单一因素，而是已经出现多因素共同推高风险的情况。",
            "Very High": "这通常意味着多个主要风险因素正在同时起作用，或者其中某些因素的强度已经足以明显抬高筛查结果。"
        }[result["risk_level"]]

        return (
            f"{intro}{meaning}"
            f" 从当前信息看，最值得关注的风险驱动包括：{detail}。"
            f" 需要特别注意的是，这个结果更适合作为“是否值得进一步检查与管理”的信号，而不是“是否已经患病”的结论。"
            f" 如果你本身已有血压、血糖、血脂或家族史方面的担忧，那么这个结果的解释价值会更偏向提醒你尽快做更客观的检测。"
        )

    elif lang == "en":
        detail = "; ".join(drivers)
        intro = {
            "Low": "The current result suggests that your risk is still broadly in the lower range.",
            "Moderate": "The current result suggests that your risk is already above a low baseline level, although this does not necessarily imply immediate disease.",
            "High": "The current result suggests that your risk profile has already accumulated enough adverse features to warrant more serious attention.",
            "Very High": "The current result suggests that, under this screening model, you fall into a distinctly elevated risk stratum."
        }[result["risk_level"]]

        meaning = {
            "Low": "This usually means the self-reported profile does not contain many strong warning signals, but it does not mean absolute safety.",
            "Moderate": "This usually means some adverse factors are already beginning to cluster, so the value of prevention and follow-up becomes more meaningful.",
            "High": "This usually means risk is being driven by multiple factors rather than a single isolated issue.",
            "Very High": "This usually means that several major risk drivers are acting at the same time, or that one or two factors are strong enough to lift the screening result substantially."
        }[result["risk_level"]]

        return (
            f"{intro} {meaning} "
            f"Based on the current inputs, the most relevant drivers include: {detail}. "
            f"It is important to interpret this result as a signal about whether more structured follow-up and objective testing may be worthwhile, rather than as proof that disease is already present. "
            f"If you already have concerns related to blood pressure, glucose, lipids, or family history, this result becomes more useful as a prompt to seek more objective evaluation."
        )

    else:
        detail = "; ".join(drivers)
        intro = {
            "Low": "El resultado actual sugiere que tu riesgo sigue estando, en términos generales, en el rango más bajo.",
            "Moderate": "El resultado actual sugiere que tu riesgo ya está por encima de un nivel basal bajo, aunque esto no implica necesariamente enfermedad inmediata.",
            "High": "El resultado actual sugiere que tu perfil de riesgo ya ha acumulado suficientes factores adversos como para requerir una atención más seria.",
            "Very High": "El resultado actual sugiere que, dentro de este modelo de cribado, perteneces a un estrato de riesgo claramente elevado."
        }[result["risk_level"]]

        meaning = {
            "Low": "Esto suele significar que el perfil autodeclarado no contiene muchas señales de alerta fuertes, pero no significa seguridad absoluta.",
            "Moderate": "Esto suele significar que algunos factores adversos ya están empezando a agruparse, por lo que la prevención y el seguimiento ganan más importancia.",
            "High": "Esto suele significar que el riesgo está siendo impulsado por varios factores y no por un único problema aislado.",
            "Very High": "Esto suele significar que varios factores principales están actuando al mismo tiempo, o que uno o dos factores tienen suficiente intensidad como para elevar claramente el resultado del cribado."
        }[result["risk_level"]]

        return (
            f"{intro} {meaning} "
            f"Según la información actual, los factores más relevantes incluyen: {detail}. "
            f"Es importante interpretar este resultado como una señal sobre si puede valer la pena un seguimiento más estructurado y pruebas objetivas, y no como una prueba de que la enfermedad ya esté presente. "
            f"Si ya tienes preocupaciones relacionadas con la presión arterial, la glucosa, los lípidos o los antecedentes familiares, este resultado resulta más útil como motivo para buscar una evaluación más objetiva."
        )

# ============================================================
# 个性化建议：文字也加长
# ============================================================
def build_segmented_advice(payload_all: dict, result: dict):
    age = payload_all["RIDAGEYR"]
    bmi = payload_all["BMXBMI"]
    smoking = payload_all["SMQ020"]
    diabetes = payload_all["DIQ010"]
    htn = payload_all["BPQ020"]
    activity = payload_all["PAQ605"]
    family = payload_all["FAMILY_HISTORY"]
    alcohol = payload_all["ALCOHOL_LEVEL"]
    level = result["risk_level"]

    adv = []

    def tr(zh, en, es):
        return {"zh": zh, "en": en, "es": es}[lang]

    if age >= 60 and htn == 1:
        adv.append(tr(
            "老年且合并高血压：你的重点不应该只是“知道自己有高血压”，而应该放在是否真正做到稳定、长期和可追踪的控制。建议把家庭血压监测、用药依从性以及医生给出的目标范围放在优先位置。",
            "Older adult with hypertension: the key issue is not simply knowing that hypertension exists, but whether it is being controlled in a stable, long-term, and trackable way. Home blood-pressure monitoring, medication adherence, and clinician-defined targets should be priorities.",
            "Adulto mayor con hipertensión: la cuestión clave no es solo saber que existe hipertensión, sino si se está controlando de forma estable, sostenida y verificable. El control domiciliario de la presión arterial, la adherencia al tratamiento y los objetivos definidos por un profesional deberían ser prioritarios."
        ))

    if age >= 60 and activity == 2:
        adv.append(tr(
            "老年且活动不足：更适合你的通常不是偶尔一次很累的运动，而是低冲击、规律、可长期持续的活动方式。你更需要的是可执行性，而不是短期强度。",
            "Older adult with low activity: what usually matters more is not occasional exhausting exercise, but low-impact, regular, and sustainable activity. Practical consistency is more valuable than short bursts of intensity.",
            "Adulto mayor con baja actividad: normalmente importa más la actividad regular, sostenible y de bajo impacto que los esfuerzos intensos ocasionales. La constancia práctica vale más que episodios breves de alta intensidad."
        ))

    if bmi >= 30 and activity == 2:
        adv.append(tr(
            "肥胖且活动不足：如果这两个因素同时存在，那么管理重点通常不应放在极端节食或短期突击，而应放在更现实的能量摄入控制、步数增加和体重缓慢下降上。",
            "Obesity with insufficient activity: when these two factors coexist, management should usually focus less on extreme dieting or short-term effort and more on realistic energy-intake control, higher daily movement, and gradual weight reduction.",
            "Obesidad con actividad insuficiente: cuando estos dos factores coexisten, la gestión debería centrarse menos en dietas extremas o esfuerzos de corto plazo y más en un control realista de la ingesta energética, mayor movimiento diario y reducción gradual del peso."
        ))

    if smoking == 1:
        adv.append(tr(
            "吸烟人群：如果只能优先做一件事，戒烟往往是最有回报的干预之一。它不仅影响单独风险，也会放大血压、代谢和炎症相关风险的长期后果。",
            "Current smoker profile: if only one intervention can be prioritized, smoking cessation is often among the highest-yield options. It affects not only direct risk, but also amplifies the long-term consequences of blood pressure, metabolic, and inflammatory burden.",
            "Perfil de fumador actual: si solo pudiera priorizarse una intervención, dejar de fumar suele estar entre las opciones de mayor impacto. No solo afecta el riesgo directo, sino que también amplifica las consecuencias a largo plazo de la carga tensional, metabólica e inflamatoria."
        ))

    if diabetes == 1:
        adv.append(tr(
            "糖尿病相关人群：你的重点不应只停留在“有没有糖尿病”，而应进一步关注血糖控制是否稳定，以及它是否同时伴随体重、血压和生活方式问题。",
            "Diabetes-related profile: the key issue is not merely whether diabetes exists, but whether glucose control is stable and whether it coexists with weight, blood pressure, and lifestyle-related issues.",
            "Perfil relacionado con diabetes: la cuestión clave no es solo si existe diabetes, sino si el control glucémico es estable y si coexiste con problemas de peso, presión arterial y estilo de vida."
        ))

    if htn == 1 and bmi >= 25:
        adv.append(tr(
            "高血压合并超重：这通常意味着风险不太可能通过单一行为改变完全逆转，更合理的目标是让体重和血压一起朝更稳定的方向改善。",
            "Hypertension plus excess weight: this often means risk is unlikely to be fully reversed through one isolated behavior change. A more realistic goal is to improve both weight and blood pressure in a coordinated way.",
            "Hipertensión con exceso de peso: esto suele significar que el riesgo difícilmente se revertirá por completo con un solo cambio conductual aislado. Un objetivo más realista es mejorar de forma coordinada tanto el peso como la presión arterial."
        ))

    if family == 1:
        adv.append(tr(
            "存在家族史：即使家族史暂时没有直接进入模型打分，它依然意味着你在解释结果时应更倾向于谨慎，而不是过度乐观。",
            "Family-history profile: even though family history does not directly enter the current model score, it still means the result should be interpreted with more caution rather than excessive reassurance.",
            "Perfil con antecedentes familiares: aunque los antecedentes familiares no entren directamente en la puntuación del modelo actual, el resultado debe interpretarse con mayor cautela y no con exceso de tranquilidad."
        ))

    if alcohol == 3:
        adv.append(tr(
            "经常饮酒：如果你已经存在其他风险因素，那么高频饮酒更值得被视为“风险放大器”，而不是单独看待的生活方式细节。",
            "Frequent alcohol intake: if other risk factors are already present, high-frequency drinking is better viewed as a risk amplifier rather than a minor lifestyle detail.",
            "Consumo frecuente de alcohol: si ya existen otros factores de riesgo, el consumo frecuente debería verse más como un amplificador del riesgo que como un simple detalle del estilo de vida."
        ))

    if age < 45 and smoking == 1:
        adv.append(tr(
            "年轻吸烟人群：年龄的确可能让短期概率看起来没那么极端，但这并不代表长期风险可以忽略。越早改变，收益通常越大。",
            "Younger smoker profile: age may make short-term probability look less extreme, but that does not mean long-term risk is negligible. Earlier change usually produces larger long-run benefit.",
            "Perfil de fumador joven: la edad puede hacer que la probabilidad a corto plazo parezca menos extrema, pero eso no significa que el riesgo a largo plazo sea despreciable. Cuanto antes se cambie, mayor suele ser el beneficio futuro."
        ))

    if age < 45 and bmi >= 30:
        adv.append(tr(
            "年轻肥胖人群：当前结果未必处于最高层级，但这类情况往往最值得在风险尚可逆时尽早介入。",
            "Younger obesity-dominant profile: the current result may not yet be in the highest tier, but this is often the stage where early intervention is most worthwhile while risk remains more reversible.",
            "Perfil joven con obesidad predominante: el resultado actual puede no estar todavía en el nivel más alto, pero esta suele ser la etapa en la que la intervención temprana es más valiosa mientras el riesgo sigue siendo más reversible."
        ))

    if diabetes == 1 and htn == 1:
        adv.append(tr(
            "糖尿病合并高血压：这是比单一风险因素更值得重视的组合，因为它往往提示风险来源不止一条路径，随访和客观检测的重要性会进一步上升。",
            "Diabetes plus hypertension: this is more important than a single isolated factor because it often implies that risk is being driven through multiple pathways, making structured follow-up and objective testing even more valuable.",
            "Diabetes más hipertensión: esto es más importante que un solo factor aislado, porque suele implicar que el riesgo está siendo impulsado por múltiples vías, lo que aumenta el valor del seguimiento estructurado y de las pruebas objetivas."
        ))

    if level == "Very High":
        adv.append(tr(
            "很高风险组：虽然这不是诊断结论，但从风险管理角度看，继续停留在自我估计阶段的价值已经有限，更合理的是尽快获得血压、血糖、血脂等客观信息。",
            "Very high screening category: although this is not a diagnostic conclusion, the value of staying only at the self-estimation stage is limited. A more reasonable next step is to obtain objective information such as blood pressure, glucose, and lipids as soon as possible.",
            "Categoría de riesgo muy alto: aunque esto no es una conclusión diagnóstica, el valor de permanecer solo en la etapa de autoestimación es limitado. Un paso más razonable es obtener cuanto antes información objetiva como presión arterial, glucosa y lípidos."
        ))

    if not adv:
        adv.append(tr(
            "当前没有触发特别明确的分层特征，这通常意味着风险管理更应落在长期维持健康习惯和规律体检上，而不是过度紧张或完全忽视。",
            "No especially strong subgroup pattern was triggered. This usually means that risk management should focus more on maintaining healthy habits and routine check-ups over time, rather than either overreacting or ignoring the issue entirely.",
            "No se activó un patrón de subgrupo especialmente fuerte. Esto suele significar que la gestión del riesgo debería centrarse más en mantener hábitos saludables y controles rutinarios a lo largo del tiempo, en lugar de reaccionar de forma exagerada o ignorar el asunto."
        ))

    return adv[:8]

# ============================================================
# 图表1
# ============================================================
def plot_risk_probability(prob: float):
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.barh(["Risk Probability"], [prob], height=0.45)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Estimated Risk Probability")
    ax.text(min(prob + 0.02, 0.92), 0, f"{prob:.3f}", va="center")
    plt.tight_layout()
    return fig

# ============================================================
# 雷达图
# 图内标签固定英文，页面标题承担三语言展示
# ============================================================
def compute_factor_scores(payload_all: dict):
    scores = {
        "Age": 0,
        "BMI": 0,
        "Smoking": 0,
        "Diabetes": 0,
        "Hypertension": 0,
        "Inactivity": 0
    }

    age = payload_all["RIDAGEYR"]
    bmi = payload_all["BMXBMI"]

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

    scores["Smoking"] = 100 if payload_all["SMQ020"] == 1 else 0
    scores["Diabetes"] = 100 if payload_all["DIQ010"] == 1 else 0
    scores["Hypertension"] = 100 if payload_all["BPQ020"] == 1 else 0
    scores["Inactivity"] = 100 if payload_all["PAQ605"] == 2 else 0

    return scores

def compute_reference_scores(payload_all: dict):
    scores = {
        "Age": 0,
        "BMI": 15,
        "Smoking": 0,
        "Diabetes": 0,
        "Hypertension": 0,
        "Inactivity": 10
    }

    age = payload_all["RIDAGEYR"]

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

def plot_small_radar_chart(scores: dict):
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

    plt.tight_layout()
    return fig

# ============================================================
# 图表3
# ============================================================
def plot_baseline_comparison(prob: float, baseline=0.045):
    labels = ["Estimated Risk", "Baseline Risk"]
    values = [prob, baseline]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(labels, values)
    ax.set_ylim(0, max(0.2, prob + 0.05))
    ax.set_ylabel("Probability")
    ax.set_title("Estimated Risk vs Baseline")

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

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(T["age"], min_value=20, max_value=100, value=45)
    sex = st.selectbox(T["sex"], options=[1, 2], format_func=lambda x: T["male"] if x == 1 else T["female"])
    height_cm = st.number_input(T["height"], min_value=120.0, max_value=220.0, value=170.0)

with col2:
    weight_kg = st.number_input(T["weight"], min_value=30.0, max_value=200.0, value=65.0)
    smoking = st.selectbox(T["smoke"], options=[1, 2], format_func=lambda x: T["yes"] if x == 1 else T["no"])
    diabetes = st.selectbox(T["diabetes"], options=[1, 2], format_func=lambda x: T["yes"] if x == 1 else T["no"])

with col3:
    hypertension = st.selectbox(T["hypertension"], options=[1, 2], format_func=lambda x: T["yes"] if x == 1 else T["no"])
    activity = st.selectbox(T["activity"], options=[1, 2], format_func=lambda x: T["yes"] if x == 1 else T["no"])
    family_history = st.selectbox(T["family_history"], options=[1, 2], format_func=lambda x: T["yes"] if x == 1 else T["no"])
    alcohol_level = st.selectbox(
        T["alcohol"],
        options=[1, 2, 3],
        format_func=lambda x: T["none"] if x == 1 else T["occasional"] if x == 2 else T["frequent"]
    )

bmi = weight_kg / ((height_cm / 100) ** 2)
st.markdown(
    f"""
    <div class="info-card">
    <b>{T["bmi_auto"]}:</b> {bmi:.2f}
    </div>
    """,
    unsafe_allow_html=True
)

payload_model = {
    "RIDAGEYR": age,
    "RIAGENDR": sex,
    "BMXBMI": bmi,
    "SMQ020": smoking,
    "DIQ010": diabetes,
    "BPQ020": hypertension,
    "PAQ605": activity
}

payload_all = {
    **payload_model,
    "FAMILY_HISTORY": family_history,
    "ALCOHOL_LEVEL": alcohol_level
}

# ============================================================
# 输出区域
# ============================================================
if st.button(T["button"]):
    result = predict_user_local(payload_model)
    risk_level_translated = translate_risk_level(result["risk_level"])
    badge_class = risk_badge_class(result["risk_level"])
    explanation = build_explanation(payload_all, result)
    advice_list = build_segmented_advice(payload_all, result)

    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="margin-top:0;">{T["result_title"]}
                <span class="risk-badge {badge_class}">{risk_level_translated}</span>
            </h3>
            <p><b>{T["probability"]}:</b> {result['risk_probability']:.3f}</p>
            <p><b>{T["risk_category"]}:</b> {risk_level_translated}</p>
            <p><b>{T["threshold"]}:</b> {result['used_threshold']:.3f}</p>
            <p><b>{T["flagged"]}:</b> {result['risk_prediction']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(T["interpretation"])
    st.write(explanation)

    st.subheader(T["how_to_read"])
    if lang == "zh":
        st.write(
            "请把这个结果理解为“在当前输入信息下，你是否更值得进一步关注”的信号，而不是最终结论。"
            " 概率越高，说明在这套筛查模型中，你与较高风险人群的相似度越高；但这仍然依赖于输入变量的范围，不能替代血压、血糖、血脂、心电图或医生判断。"
            " 同时，由于这是一个低患病率场景，模型更适合帮助识别谁更值得后续检查，而不是直接断言谁已经患病。"
        )
    elif lang == "en":
        st.write(
            "Please interpret this result as a signal of whether you may be more worthy of further attention under the current self-reported profile, rather than as a final conclusion. "
            "A higher probability means stronger similarity to higher-risk groups within this screening model, but it still depends on the limited input variables and does not replace blood pressure, glucose, lipids, ECG, or physician judgment. "
            "Because this is a low-prevalence setting, the model is better suited to flagging who may deserve follow-up rather than declaring who definitely has disease."
        )
    else:
        st.write(
            "Interpreta este resultado como una señal de si mereces una atención adicional según tu perfil autodeclarado actual, y no como una conclusión final. "
            "Una probabilidad más alta significa una mayor similitud con grupos de mayor riesgo dentro de este modelo de cribado, pero sigue dependiendo de variables limitadas y no sustituye la presión arterial, la glucosa, los lípidos, el ECG ni la valoración médica. "
            "Dado que se trata de un contexto de baja prevalencia, el modelo es más útil para señalar quién merece seguimiento que para afirmar quién tiene definitivamente enfermedad."
        )

    st.subheader(T["chart1_title"])
    fig1 = plot_risk_probability(result["risk_probability"])
    st.pyplot(fig1)

    st.subheader(T["chart2_title"])
    user_scores = compute_factor_scores(payload_all)
    ref_scores = compute_reference_scores(payload_all)

    radar_col1, radar_col2 = st.columns(2)
    with radar_col1:
        st.markdown(f"**{T['your_profile']}**")
        fig_user = plot_small_radar_chart(user_scores)
        st.pyplot(fig_user)

    with radar_col2:
        st.markdown(f"**{T['reference_profile']}**")
        fig_ref = plot_small_radar_chart(ref_scores)
        st.pyplot(fig_ref)

    st.subheader(T["chart3_title"])
    fig3 = plot_baseline_comparison(result["risk_probability"], baseline=0.045)
    st.pyplot(fig3)

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
footer_html = """
<hr style="margin-top: 2.5rem; margin-bottom: 1rem; border: none; border-top: 1px solid #d1d5db;">

<div style="text-align: center; font-size: 0.98rem; color: #4a5568; margin-bottom: 0.35rem;">
    <strong>Author:</strong> Zikai "Constantin" Wang
</div>

<div style="text-align: center; font-size: 0.92rem; color: #718096; margin-bottom: 1.1rem;">
    <strong>Data Source:</strong> U.S. Centers for Disease Control and Prevention (CDC), NHANES datasets
</div>

<div style="
    margin-top: 0.8rem;
    padding: 1rem 1.1rem;
    border-radius: 14px;
    background-color: #f8fafc;
    color: #4a5568;
    font-size: 0.92rem;
    line-height: 1.7;
">
    <div style="font-weight: 700; margin-bottom: 0.55rem;">Disclaimer</div>

    <div style="margin-bottom: 0.75rem;">
        This application is intended for <strong>educational and screening purposes only</strong>.
        It does not provide medical advice, diagnosis, or treatment.
        The estimated results are generated from statistical and machine learning models trained on publicly available data,
        and should be interpreted as <strong>probabilistic screening signals rather than clinical conclusions</strong>.
    </div>

    <div style="margin-bottom: 0.45rem;">
        Users should understand that:
    </div>

    <ul style="margin-top: 0.2rem; margin-bottom: 0.85rem; padding-left: 1.2rem;">
        <li>The model relies on self-reported or simplified input variables and may not fully capture an individual’s actual health condition.</li>
        <li>The output is subject to model limitations, data bias, sampling constraints, and generalization error.</li>
        <li>A low predicted risk does not guarantee absence of disease, and a high predicted risk does not confirm diagnosis.</li>
    </ul>

    <div style="margin-bottom: 0.75rem;">
        If you are experiencing symptoms such as chest pain, shortness of breath, dizziness, fainting,
        or any other concerning physical condition, please seek evaluation from a
        <strong>qualified healthcare professional</strong> as soon as possible.
    </div>

    <div>
        By using this tool, you acknowledge that the developer and data providers bear
        <strong>no responsibility for medical, financial, or personal decisions</strong>
        made on the basis of the output.
    </div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)