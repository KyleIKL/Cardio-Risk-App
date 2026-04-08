
# 1. 导入库

import pandas as pd
import os


# 2. 设置数据路径（你的文件夹）

DATA_PATH = r"E:\Project99\2017-2018"   # ← 改成你实际路径


# 3. 定义读取函数（通用）

def load_xpt(file_name):
    """
    读取 NHANES XPT 文件
    参数:
        file_name: 文件名
    返回:
        pandas DataFrame
    """
    file_path = os.path.join(DATA_PATH, file_name)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_name} 不存在")
    
    # 读取 XPT
    df = pd.read_sas(file_path)
    
    # 列名统一转大写
    df.columns = df.columns.str.upper()
    
    return df



# 4. 加载所有数据


# 人口学
demo = load_xpt("DEMO_J.xpt")

# 问卷
mcq = load_xpt("MCQ_J.xpt")     # 冠心病
diq = load_xpt("DIQ_J.xpt")     # 糖尿病
bpq = load_xpt("BPQ_J.xpt")     # 高血压
smq = load_xpt("SMQ_J.xpt")     # 吸烟
alq = load_xpt("ALQ_J.xpt")     # 饮酒
paq = load_xpt("PAQ_J.xpt")     # 运动
inq = load_xpt("INQ_J.xpt")     # 收入
hsq = load_xpt("HSQ_J.xpt")     # 自评健康
hiq = load_xpt("HIQ_J.xpt")     # 医保

# 体检数据
bmx = load_xpt("BMX_J.xpt")     # BMI
bpx = load_xpt("BPX_J.xpt")     # 血压

# 实验室数据
hdl = load_xpt("HDL_J.xpt")    # 高密度脂蛋白(负相关)
tchol = load_xpt("TCHOL_J.xpt") # 总胆固醇(正相关)
trigly = load_xpt("TRIGLY_J.xpt") # 甘油三酯(强正相关)
glu = load_xpt("GLU_J.xpt")     # 血糖(风险直接体现)
ghb = load_xpt("GHB_J.xpt")     # 糖化血红蛋白(稳定风险体现)
hscrp = load_xpt("HSCRP_J.xpt") # 敏感C反应蛋白(判断炎症水平)
ins = load_xpt("INS_J.xpt")     # 胰岛素(胰岛素抵抗体现)
cbc = load_xpt("CBC_J.xpt")     # 血液检查(可选)
alb_cr = load_xpt("ALB_CR_J.xpt")  # 尿白蛋白/肌酐比值(可选)
lux = load_xpt("LUX_J.xpt")     # 可选(肝脂)


# 5. 定义 merge 函数

def merge_data(base, df_list):
    """
    按 SEQN 合并所有表
    参数:
        base: 主表（一般 DEMO）
        df_list: 其他表列表
    """
    merged = base.copy()
    
    for df in df_list:
        merged = pd.merge(
            merged,
            df,
            on="SEQN",
            how="left"  # 保留主样本
        )
    
    return merged



# 6. 执行合并

df = merge_data(
    demo,
    [
        mcq, diq, bpq, smq, alq, paq, inq, hsq, hiq,
        bmx, bpx,
        hdl, tchol, trigly,
        glu, ghb, hscrp,
        ins, cbc, alb_cr, lux
    ]
)


# 7. 构造标签（CHD）

# MCQ160C: 是否被诊断冠心病
df["CHD"] = df["MCQ160C"].map({
    1: 1,   # Yes
    2: 0    # No
})


# 8. 基础筛选

# 只保留成年人
df = df[df["RIDAGEYR"] >= 20]

# 删除标签缺失
df = df[df["CHD"].notna()]


# 9. 常用变量构造


# BMI
df["BMI"] = df["BMXBMI"]

# 收缩压（取平均）
df["SBP"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)

# 舒张压
df["DBP"] = df[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)


# 10. 查看数据

print(df.shape)
print(df[["CHD", "RIDAGEYR", "BMI", "SBP", "LBXTC"]].head())


df_user = df[[
    "CHD",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "SMQ020",
    "DIQ010",
    "BPQ020",
    "PAQ605"
]].dropna()

df_full = df[[
    "CHD",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "SBP",
    "LBXTC",
    "LBXTR",
    "LBDHDD",
    "LBXGLU",   # 保留
    "LBXGH",    # 保留
]].dropna()

print(df_user.shape)
print(df_full.shape)
print(df_user.head())
print(df_full.head())

# 训练部分

# ============================================================
# 0. 导入库
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. 全局配置
# ============================================================
RANDOM_STATE = 42
N_JOBS = -1

# 你已经准备好的数据：
# df_user
# df_full

# ============================================================
# 2. 基础函数
# ============================================================
def split_xy(df: pd.DataFrame, target_col: str = "CHD"):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()
    return X, y


def make_train_valid_test_split(X, y, test_size=0.2, valid_size=0.2, random_state=42):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full,
        test_size=valid_size,
        stratify=y_train_full,
        random_state=random_state
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_preprocessor_for_logistic(X: pd.DataFrame):
    numeric_features = X.columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="drop"
    )
    return preprocessor


def build_preprocessor_for_tree(X: pd.DataFrame):
    numeric_features = X.columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="drop"
    )
    return preprocessor


def compute_sample_weight_binary(y):
    y = pd.Series(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()

    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y), dtype=float)

    w_pos = n_neg / n_pos
    weights = np.where(y == 1, w_pos, 1.0)
    return weights


def print_dataset_summary(name, X_train, X_valid, X_test, y_train, y_valid, y_test):
    print("=" * 90)
    print(f"{name} 数据概况")
    print(f"Train shape: {X_train.shape}, positive rate: {y_train.mean():.4f}")
    print(f"Valid shape: {X_valid.shape}, positive rate: {y_valid.mean():.4f}")
    print(f"Test  shape: {X_test.shape}, positive rate: {y_test.mean():.4f}")


# ============================================================
# 3. 阈值选择：改成 precision 导向
# ============================================================
def select_threshold_by_precision_constraint(
    y_true,
    y_prob,
    min_precision=0.30,
    min_recall=0.15
):
    """
    在验证集上选阈值：
    1. 优先满足 precision >= min_precision
    2. 同时 recall >= min_recall
    3. 在满足条件中选择 F1 最高者
    4. 若没有满足条件的，则退而求其次选 F1 最高
    """
    thresholds = np.linspace(0.05, 0.95, 181)

    best_thr = 0.5
    best_score = -1
    found_feasible = False

    y_true = np.asarray(y_true)

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        feasible = (precision >= min_precision) and (recall >= min_recall)

        if feasible:
            if (not found_feasible) or (f1 > best_score):
                best_thr = thr
                best_score = f1
                found_feasible = True

        elif not found_feasible:
            # 如果尚未找到满足 precision 约束的解，则先记住 F1 最好的
            if f1 > best_score:
                best_thr = thr
                best_score = f1

    return best_thr, best_score, found_feasible


def risk_level_from_probability(prob):
    """
    风险分层：更适合产品展示
    """
    if prob < 0.05:
        return "Low"
    elif prob < 0.15:
        return "Moderate"
    elif prob < 0.30:
        return "High"
    else:
        return "Very High"


def evaluate_predictions(y_true, y_prob, threshold, name="model"):
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("=" * 90)
    print(f"模型: {name}")
    print(f"使用阈值: {threshold:.3f}")
    print(f"ROC AUC : {auc:.4f}")
    print(f"AP      : {ap:.4f}")
    print(f"Precision(1): {precision:.4f}")
    print(f"Recall(1)   : {recall:.4f}")
    print(f"F1(1)       : {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "auc": auc,
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold
    }


def save_curves(y_true, y_prob, prefix="model"):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {prefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_roc.png", dpi=200)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {prefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_pr.png", dpi=200)
    plt.close()


# ============================================================
# 4. 用户模型：Logistic
# ============================================================
def train_user_logit(df_user, min_precision=0.30, min_recall=0.20):
    X, y = split_xy(df_user, "CHD")
    X_train, X_valid, X_test, y_train, y_valid, y_test = make_train_valid_test_split(
        X, y,
        test_size=0.2,
        valid_size=0.2,
        random_state=RANDOM_STATE
    )

    print_dataset_summary("用户模型 / Logistic", X_train, X_valid, X_test, y_train, y_valid, y_test)

    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor_for_logistic(X_train)),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE
        ))
    ])

    param_dist = {
        "model__C": np.logspace(-3, 1, 12)
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=8,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    print("\n[user_logit] 最佳参数：")
    print(search.best_params_)
    print("[user_logit] 最佳CV AP：", round(search.best_score_, 4))

    best_model = search.best_estimator_

    valid_prob = best_model.predict_proba(X_valid)[:, 1]
    test_prob = best_model.predict_proba(X_test)[:, 1]

    best_thr, best_score, feasible = select_threshold_by_precision_constraint(
        y_true=y_valid.values,
        y_prob=valid_prob,
        min_precision=min_precision,
        min_recall=min_recall
    )

    print(f"[user_logit] 验证集最优阈值: {best_thr:.3f}")
    print(f"[user_logit] 阈值是否满足precision约束: {feasible}")
    print(f"[user_logit] 选择分数(F1): {best_score:.4f}")

    metrics = evaluate_predictions(y_test.values, test_prob, best_thr, name="user_logit")
    save_curves(y_test.values, test_prob, prefix="user_logit")

    return {
        "name": "user_logit",
        "model": best_model,
        "threshold": best_thr,
        "metrics": metrics,
        "features": X.columns.tolist()
    }


# ============================================================
# 5. 用户模型：HGB
# ============================================================
def train_user_hgb(df_user, min_precision=0.30, min_recall=0.20):
    X, y = split_xy(df_user, "CHD")
    X_train, X_valid, X_test, y_train, y_valid, y_test = make_train_valid_test_split(
        X, y,
        test_size=0.2,
        valid_size=0.2,
        random_state=RANDOM_STATE
    )

    print_dataset_summary("用户模型 / HGB", X_train, X_valid, X_test, y_train, y_valid, y_test)

    pre = build_preprocessor_for_tree(X_train)

    X_train_t = pre.fit_transform(X_train)
    X_valid_t = pre.transform(X_valid)
    X_test_t = pre.transform(X_test)

    sample_weight = compute_sample_weight_binary(y_train)

    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    param_dist = {
        "max_iter": [80, 100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [2, 3],
        "min_samples_leaf": [10, 20, 30, 40],
        "l2_regularization": [0.0, 0.1, 1.0, 5.0]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        refit=True
    )

    search.fit(X_train_t, y_train, sample_weight=sample_weight)

    print("\n[user_hgb] 最佳参数：")
    print(search.best_params_)
    print("[user_hgb] 最佳CV AP：", round(search.best_score_, 4))

    best_model = search.best_estimator_
    best_model.fit(X_train_t, y_train, sample_weight=sample_weight)

    valid_prob = best_model.predict_proba(X_valid_t)[:, 1]
    test_prob = best_model.predict_proba(X_test_t)[:, 1]

    best_thr, best_score, feasible = select_threshold_by_precision_constraint(
        y_true=y_valid.values,
        y_prob=valid_prob,
        min_precision=min_precision,
        min_recall=min_recall
    )

    print(f"[user_hgb] 验证集最优阈值: {best_thr:.3f}")
    print(f"[user_hgb] 阈值是否满足precision约束: {feasible}")
    print(f"[user_hgb] 选择分数(F1): {best_score:.4f}")

    metrics = evaluate_predictions(y_test.values, test_prob, best_thr, name="user_hgb")
    save_curves(y_test.values, test_prob, prefix="user_hgb")

    return {
        "name": "user_hgb",
        "model": best_model,
        "preprocessor": pre,
        "threshold": best_thr,
        "metrics": metrics,
        "features": X.columns.tolist()
    }


# ============================================================
# 6. 完整模型：Logistic
# ============================================================
def train_full_logit(df_full, min_precision=0.25, min_recall=0.20):
    X, y = split_xy(df_full, "CHD")
    X_train, X_valid, X_test, y_train, y_valid, y_test = make_train_valid_test_split(
        X, y,
        test_size=0.2,
        valid_size=0.2,
        random_state=RANDOM_STATE
    )

    print_dataset_summary("完整模型 / Logistic", X_train, X_valid, X_test, y_train, y_valid, y_test)

    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor_for_logistic(X_train)),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE
        ))
    ])

    param_dist = {
        "model__C": np.logspace(-3, 1, 12)
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=8,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    print("\n[full_logit] 最佳参数：")
    print(search.best_params_)
    print("[full_logit] 最佳CV AP：", round(search.best_score_, 4))

    best_model = search.best_estimator_

    valid_prob = best_model.predict_proba(X_valid)[:, 1]
    test_prob = best_model.predict_proba(X_test)[:, 1]

    best_thr, best_score, feasible = select_threshold_by_precision_constraint(
        y_true=y_valid.values,
        y_prob=valid_prob,
        min_precision=min_precision,
        min_recall=min_recall
    )

    print(f"[full_logit] 验证集最优阈值: {best_thr:.3f}")
    print(f"[full_logit] 阈值是否满足precision约束: {feasible}")
    print(f"[full_logit] 选择分数(F1): {best_score:.4f}")

    metrics = evaluate_predictions(y_test.values, test_prob, best_thr, name="full_logit")
    save_curves(y_test.values, test_prob, prefix="full_logit")

    return {
        "name": "full_logit",
        "model": best_model,
        "threshold": best_thr,
        "metrics": metrics,
        "features": X.columns.tolist()
    }


# ============================================================
# 7. 完整模型：HGB
# ============================================================
def train_full_hgb(df_full, min_precision=0.25, min_recall=0.20):
    X, y = split_xy(df_full, "CHD")
    X_train, X_valid, X_test, y_train, y_valid, y_test = make_train_valid_test_split(
        X, y,
        test_size=0.2,
        valid_size=0.2,
        random_state=RANDOM_STATE
    )

    print_dataset_summary("完整模型 / HGB", X_train, X_valid, X_test, y_train, y_valid, y_test)

    pre = build_preprocessor_for_tree(X_train)

    X_train_t = pre.fit_transform(X_train)
    X_valid_t = pre.transform(X_valid)
    X_test_t = pre.transform(X_test)

    sample_weight = compute_sample_weight_binary(y_train)

    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    param_dist = {
        "max_iter": [80, 100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [2, 3],
        "min_samples_leaf": [10, 20, 30, 40],
        "l2_regularization": [0.0, 0.1, 1.0, 5.0]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        refit=True
    )

    search.fit(X_train_t, y_train, sample_weight=sample_weight)

    print("\n[full_hgb] 最佳参数：")
    print(search.best_params_)
    print("[full_hgb] 最佳CV AP：", round(search.best_score_, 4))

    best_model = search.best_estimator_
    best_model.fit(X_train_t, y_train, sample_weight=sample_weight)

    valid_prob = best_model.predict_proba(X_valid_t)[:, 1]
    test_prob = best_model.predict_proba(X_test_t)[:, 1]

    best_thr, best_score, feasible = select_threshold_by_precision_constraint(
        y_true=y_valid.values,
        y_prob=valid_prob,
        min_precision=min_precision,
        min_recall=min_recall
    )

    print(f"[full_hgb] 验证集最优阈值: {best_thr:.3f}")
    print(f"[full_hgb] 阈值是否满足precision约束: {feasible}")
    print(f"[full_hgb] 选择分数(F1): {best_score:.4f}")

    metrics = evaluate_predictions(y_test.values, test_prob, best_thr, name="full_hgb")
    save_curves(y_test.values, test_prob, prefix="full_hgb")

    return {
        "name": "full_hgb",
        "model": best_model,
        "preprocessor": pre,
        "threshold": best_thr,
        "metrics": metrics,
        "features": X.columns.tolist()
    }


# ============================================================
# 8. 主训练流程
# ============================================================
print("\n开始训练用户模型...")
user_logit_result = train_user_logit(df_user, min_precision=0.30, min_recall=0.20)
user_hgb_result = train_user_hgb(df_user, min_precision=0.30, min_recall=0.20)

print("\n开始训练完整模型...")
full_logit_result = train_full_logit(df_full, min_precision=0.25, min_recall=0.20)
full_hgb_result = train_full_hgb(df_full, min_precision=0.25, min_recall=0.20)


# ============================================================
# 9. 汇总比较
# 优先级：
# 1. precision
# 2. AP
# 3. AUC
# ============================================================
summary = pd.DataFrame([
    {
        "model": user_logit_result["name"],
        "auc": user_logit_result["metrics"]["auc"],
        "ap": user_logit_result["metrics"]["ap"],
        "precision": user_logit_result["metrics"]["precision"],
        "recall": user_logit_result["metrics"]["recall"],
        "f1": user_logit_result["metrics"]["f1"],
        "threshold": user_logit_result["threshold"]
    },
    {
        "model": user_hgb_result["name"],
        "auc": user_hgb_result["metrics"]["auc"],
        "ap": user_hgb_result["metrics"]["ap"],
        "precision": user_hgb_result["metrics"]["precision"],
        "recall": user_hgb_result["metrics"]["recall"],
        "f1": user_hgb_result["metrics"]["f1"],
        "threshold": user_hgb_result["threshold"]
    },
    {
        "model": full_logit_result["name"],
        "auc": full_logit_result["metrics"]["auc"],
        "ap": full_logit_result["metrics"]["ap"],
        "precision": full_logit_result["metrics"]["precision"],
        "recall": full_logit_result["metrics"]["recall"],
        "f1": full_logit_result["metrics"]["f1"],
        "threshold": full_logit_result["threshold"]
    },
    {
        "model": full_hgb_result["name"],
        "auc": full_hgb_result["metrics"]["auc"],
        "ap": full_hgb_result["metrics"]["ap"],
        "precision": full_hgb_result["metrics"]["precision"],
        "recall": full_hgb_result["metrics"]["recall"],
        "f1": full_hgb_result["metrics"]["f1"],
        "threshold": full_hgb_result["threshold"]
    }
]).sort_values(["precision", "ap", "auc"], ascending=False)

print("\n模型总对比：")
print(summary)


# ============================================================
# 10. 各任务内部选最终模型
# 用户模型：先比 precision，再比 ap
# 完整模型：先比 precision，再比 ap
# ============================================================
user_candidates = pd.DataFrame([
    {
        "name": user_logit_result["name"],
        "precision": user_logit_result["metrics"]["precision"],
        "ap": user_logit_result["metrics"]["ap"]
    },
    {
        "name": user_hgb_result["name"],
        "precision": user_hgb_result["metrics"]["precision"],
        "ap": user_hgb_result["metrics"]["ap"]
    }
]).sort_values(["precision", "ap"], ascending=False)

final_user_name = user_candidates.iloc[0]["name"]
final_user_result = user_logit_result if final_user_name == "user_logit" else user_hgb_result

full_candidates = pd.DataFrame([
    {
        "name": full_logit_result["name"],
        "precision": full_logit_result["metrics"]["precision"],
        "ap": full_logit_result["metrics"]["ap"]
    },
    {
        "name": full_hgb_result["name"],
        "precision": full_hgb_result["metrics"]["precision"],
        "ap": full_hgb_result["metrics"]["ap"]
    }
]).sort_values(["precision", "ap"], ascending=False)

final_full_name = full_candidates.iloc[0]["name"]
final_full_result = full_logit_result if final_full_name == "full_logit" else full_hgb_result

print("\n最终选择的用户模型：", final_user_result["name"])
print("最终选择的完整模型：", final_full_result["name"])


# ============================================================
# 11. 保存模型
# ============================================================
# 用户模型
if final_user_result["name"] == "user_logit":
    joblib.dump(final_user_result["model"], "final_user_model.joblib")
    joblib.dump({"threshold": final_user_result["threshold"]}, "final_user_threshold.joblib")
    joblib.dump({"features": final_user_result["features"]}, "final_user_features.joblib")
else:
    joblib.dump(final_user_result["model"], "final_user_model.joblib")
    joblib.dump(final_user_result["preprocessor"], "final_user_preprocessor.joblib")
    joblib.dump({"threshold": final_user_result["threshold"]}, "final_user_threshold.joblib")
    joblib.dump({"features": final_user_result["features"]}, "final_user_features.joblib")

# 完整模型
if final_full_result["name"] == "full_logit":
    joblib.dump(final_full_result["model"], "final_full_model.joblib")
    joblib.dump({"threshold": final_full_result["threshold"]}, "final_full_threshold.joblib")
    joblib.dump({"features": final_full_result["features"]}, "final_full_features.joblib")
else:
    joblib.dump(final_full_result["model"], "final_full_model.joblib")
    joblib.dump(final_full_result["preprocessor"], "final_full_preprocessor.joblib")
    joblib.dump({"threshold": final_full_result["threshold"]}, "final_full_threshold.joblib")
    joblib.dump({"features": final_full_result["features"]}, "final_full_features.joblib")

print("\n模型已保存完成。")


# ============================================================
# 12. 预测函数（支持风险等级）
# ============================================================
def predict_with_saved_logit(sample_df: pd.DataFrame, model_path: str, threshold_path: str):
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)["threshold"]

    prob = model.predict_proba(sample_df)[:, 1]
    pred = (prob >= threshold).astype(int)

    result = sample_df.copy()
    result["risk_probability"] = prob
    result["risk_level"] = [risk_level_from_probability(p) for p in prob]
    result["risk_prediction"] = pred
    result["used_threshold"] = threshold
    return result


def predict_with_saved_hgb(sample_df: pd.DataFrame, model_path: str, preprocessor_path: str, threshold_path: str):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    threshold = joblib.load(threshold_path)["threshold"]

    X_t = preprocessor.transform(sample_df)
    prob = model.predict_proba(X_t)[:, 1]
    pred = (prob >= threshold).astype(int)

    result = sample_df.copy()
    result["risk_probability"] = prob
    result["risk_level"] = [risk_level_from_probability(p) for p in prob]
    result["risk_prediction"] = pred
    result["used_threshold"] = threshold
    return result


# ============================================================
# 13. 示例预测
# ============================================================
user_feature_names = split_xy(df_user)[0].columns.tolist()
full_feature_names = split_xy(df_full)[0].columns.tolist()

new_user_sample = pd.DataFrame([{col: 0 for col in user_feature_names}])
new_full_sample = pd.DataFrame([{col: 0 for col in full_feature_names}])

print("\n用户模型示例预测：")
if final_user_result["name"] == "user_logit":
    pred_user = predict_with_saved_logit(
        sample_df=new_user_sample,
        model_path="final_user_model.joblib",
        threshold_path="final_user_threshold.joblib"
    )
else:
    pred_user = predict_with_saved_hgb(
        sample_df=new_user_sample,
        model_path="final_user_model.joblib",
        preprocessor_path="final_user_preprocessor.joblib",
        threshold_path="final_user_threshold.joblib"
    )
print(pred_user)

print("\n完整模型示例预测：")
if final_full_result["name"] == "full_logit":
    pred_full = predict_with_saved_logit(
        sample_df=new_full_sample,
        model_path="final_full_model.joblib",
        threshold_path="final_full_threshold.joblib"
    )
else:
    pred_full = predict_with_saved_hgb(
        sample_df=new_full_sample,
        model_path="final_full_model.joblib",
        preprocessor_path="final_full_preprocessor.joblib",
        threshold_path="final_full_threshold.joblib"
    )
print(pred_full)

# 
# ============================================================
# 10. 概率校准 + 阈值优化 + 保存最终模型
# ============================================================

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ----------------------------
# 风险分层函数
# ----------------------------
def risk_level_from_probability(prob: float) -> str:
    """
    针对当前低患病率场景的风险分层
    """
    if prob < 0.03:
        return "Low"
    elif prob < 0.08:
        return "Moderate"
    elif prob < 0.15:
        return "High"
    else:
        return "Very High"


# ----------------------------
# precision导向阈值搜索
# ----------------------------
def select_threshold_by_precision(
    y_true,
    y_prob,
    min_precision=0.20,
    min_recall=0.20
):
    thresholds = np.linspace(0.01, 0.99, 197)

    best_thr = 0.5
    best_f1 = -1
    found_feasible = False

    y_true = np.asarray(y_true)

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        feasible = (precision >= min_precision) and (recall >= min_recall)

        if feasible:
            if (not found_feasible) or (f1 > best_f1):
                best_thr = thr
                best_f1 = f1
                found_feasible = True
        elif not found_feasible:
            if f1 > best_f1:
                best_thr = thr
                best_f1 = f1

    return best_thr, best_f1, found_feasible


# ----------------------------
# 评估函数
# ----------------------------
def evaluate_calibrated_result(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("=" * 90)
    print(f"模型: {name}")
    print(f"使用阈值: {threshold:.3f}")
    print(f"ROC AUC : {auc:.4f}")
    print(f"AP      : {ap:.4f}")
    print(f"Brier   : {brier:.4f}")
    print(f"Precision(1): {precision:.4f}")
    print(f"Recall(1)   : {recall:.4f}")
    print(f"F1(1)       : {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "auc": auc,
        "ap": ap,
        "brier": brier,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold
    }


# ============================================================
# 11. 重新切分数据，用于最终校准
# ============================================================

# ----------------------------
# 用户模型数据
# ----------------------------
X_user, y_user = split_xy(df_user, "CHD")
X_user_train, X_user_valid, X_user_test, y_user_train, y_user_valid, y_user_test = make_train_valid_test_split(
    X_user, y_user,
    test_size=0.2,
    valid_size=0.2,
    random_state=42
)

# ----------------------------
# 完整模型数据
# ----------------------------
X_full, y_full = split_xy(df_full, "CHD")
X_full_train, X_full_valid, X_full_test, y_full_train, y_full_valid, y_full_test = make_train_valid_test_split(
    X_full, y_full,
    test_size=0.2,
    valid_size=0.2,
    random_state=42
)


# ============================================================
# 12. 用户模型：最终版本（Logistic + 校准）
# ============================================================

# 这里沿用你前面筛出来的 user_logit 参数
final_user_pipeline = Pipeline(steps=[
    ("preprocessor", build_preprocessor_for_logistic(X_user_train)),
    ("model", LogisticRegression(
        C=0.38986037025490716,
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ))
])

final_user_pipeline.fit(X_user_train, y_user_train)

calibrated_user_model = CalibratedClassifierCV(
    estimator=final_user_pipeline,
    method="sigmoid",
    cv=3
)
calibrated_user_model.fit(X_user_train, y_user_train)

user_valid_prob = calibrated_user_model.predict_proba(X_user_valid)[:, 1]
user_test_prob = calibrated_user_model.predict_proba(X_user_test)[:, 1]

user_best_thr, user_best_f1, user_feasible = select_threshold_by_precision(
    y_true=y_user_valid.values,
    y_prob=user_valid_prob,
    min_precision=0.20,
    min_recall=0.20
)

print("\n[最终用户模型] 验证集最优阈值:", round(user_best_thr, 3))
print("[最终用户模型] 是否满足 precision 约束:", user_feasible)
print("[最终用户模型] F1:", round(user_best_f1, 4))

user_final_metrics = evaluate_calibrated_result(
    name="final_user_logit_calibrated",
    y_true=y_user_test.values,
    y_prob=user_test_prob,
    threshold=user_best_thr
)


# ============================================================
# 13. 完整模型：最终版本（HGB + 校准）
# ============================================================

# 这里沿用你前面筛出来的 full_hgb 参数
full_preprocessor = build_preprocessor_for_tree(X_full_train)

X_full_train_t = full_preprocessor.fit_transform(X_full_train)
X_full_valid_t = full_preprocessor.transform(X_full_valid)
X_full_test_t = full_preprocessor.transform(X_full_test)

full_sample_weight = compute_sample_weight_binary(y_full_train)

final_full_model_raw = HistGradientBoostingClassifier(
    min_samples_leaf=10,
    max_iter=200,
    max_depth=2,
    learning_rate=0.05,
    l2_regularization=5.0,
    random_state=42
)

final_full_model_raw.fit(X_full_train_t, y_full_train, sample_weight=full_sample_weight)

calibrated_full_model = CalibratedClassifierCV(
    estimator=final_full_model_raw,
    method="sigmoid",
    cv=3
)
calibrated_full_model.fit(X_full_train_t, y_full_train)

full_valid_prob = calibrated_full_model.predict_proba(X_full_valid_t)[:, 1]
full_test_prob = calibrated_full_model.predict_proba(X_full_test_t)[:, 1]

full_best_thr, full_best_f1, full_feasible = select_threshold_by_precision(
    y_true=y_full_valid.values,
    y_prob=full_valid_prob,
    min_precision=0.18,
    min_recall=0.20
)

print("\n[最终完整模型] 验证集最优阈值:", round(full_best_thr, 3))
print("[最终完整模型] 是否满足 precision 约束:", full_feasible)
print("[最终完整模型] F1:", round(full_best_f1, 4))

full_final_metrics = evaluate_calibrated_result(
    name="final_full_hgb_calibrated",
    y_true=y_full_test.values,
    y_prob=full_test_prob,
    threshold=full_best_thr
)


# ============================================================
# 14. 保存模型与配置
# ============================================================

joblib.dump(calibrated_user_model, "calibrated_user_model.joblib")
joblib.dump({"threshold": user_best_thr}, "calibrated_user_threshold.joblib")
joblib.dump({"features": X_user.columns.tolist()}, "calibrated_user_features.joblib")

joblib.dump(calibrated_full_model, "calibrated_full_model.joblib")
joblib.dump(full_preprocessor, "calibrated_full_preprocessor.joblib")
joblib.dump({"threshold": full_best_thr}, "calibrated_full_threshold.joblib")
joblib.dump({"features": X_full.columns.tolist()}, "calibrated_full_features.joblib")

print("\n校准后的最终模型已保存完成。")


# ============================================================
# 15. 本地预测接口函数
# 这部分后面可直接给网站 API 调用
# ============================================================

def predict_user_risk_local(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    用户模型本地预测
    输入：DataFrame，列必须与训练时一致
    输出：附加风险概率、风险等级、是否超过阈值
    """
    model = joblib.load("calibrated_user_model.joblib")
    threshold = joblib.load("calibrated_user_threshold.joblib")["threshold"]

    prob = model.predict_proba(sample_df)[:, 1]
    pred = (prob >= threshold).astype(int)

    result = sample_df.copy()
    result["risk_probability"] = prob
    result["risk_level"] = [risk_level_from_probability(float(p)) for p in prob]
    result["risk_prediction"] = pred
    result["used_threshold"] = threshold
    return result


def predict_full_risk_local(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    完整模型本地预测
    """
    model = joblib.load("calibrated_full_model.joblib")
    preprocessor = joblib.load("calibrated_full_preprocessor.joblib")
    threshold = joblib.load("calibrated_full_threshold.joblib")["threshold"]

    X_t = preprocessor.transform(sample_df)
    prob = model.predict_proba(X_t)[:, 1]
    pred = (prob >= threshold).astype(int)

    result = sample_df.copy()
    result["risk_probability"] = prob
    result["risk_level"] = [risk_level_from_probability(float(p)) for p in prob]
    result["risk_prediction"] = pred
    result["used_threshold"] = threshold
    return result


# ============================================================
# 16. 预留正式网站接口主体
# 这里先写成 Python 函数，后面可平移到 FastAPI / Flask
# ============================================================

def website_api_predict_user(payload: dict) -> dict:
    """
    正式网站的用户模型接口主体（占位版）
    后面你可以把这个函数搬到 FastAPI/Flask 的 POST 路由里
    """
    sample_df = pd.DataFrame([payload])
    result = predict_user_risk_local(sample_df)

    return {
        "risk_probability": float(result["risk_probability"].iloc[0]),
        "risk_level": str(result["risk_level"].iloc[0]),
        "risk_prediction": int(result["risk_prediction"].iloc[0]),
        "used_threshold": float(result["used_threshold"].iloc[0]),
        "message": (
            "This tool is for cardiovascular risk screening only and does not constitute a medical diagnosis."
        )
    }


def website_api_predict_full(payload: dict) -> dict:
    """
    正式网站的完整模型接口主体（占位版）
    """
    sample_df = pd.DataFrame([payload])
    result = predict_full_risk_local(sample_df)

    return {
        "risk_probability": float(result["risk_probability"].iloc[0]),
        "risk_level": str(result["risk_level"].iloc[0]),
        "risk_prediction": int(result["risk_prediction"].iloc[0]),
        "used_threshold": float(result["used_threshold"].iloc[0]),
        "message": (
            "This tool is for cardiovascular risk screening only and does not constitute a medical diagnosis."
        )
    }


# ============================================================
# 17. 示例调用
# ============================================================

print("\n用户模型示例调用：")
user_payload_example = {
    "RIDAGEYR": 55,
    "RIAGENDR": 1,
    "BMXBMI": 28.0,
    "SMQ020": 1,
    "DIQ010": 2,
    "BPQ020": 1,
    "PAQ605": 2
}
print(website_api_predict_user(user_payload_example))

print("\n完整模型示例调用：")
full_payload_example = {
    "RIDAGEYR": 55,
    "RIAGENDR": 1,
    "BMXBMI": 28.0,
    "SBP": 145,
    "LBXTC": 220,
    "LBXTR": 180,
    "LBDHDD": 38,
    "LBXGLU": 110,
    "LBXGH": 6.0
}
print(website_api_predict_full(full_payload_example))