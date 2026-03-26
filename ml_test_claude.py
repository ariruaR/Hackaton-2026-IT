# bot_detection_ml.py
# ============================================================
# ML-модель детекта ботов.
# Супервизор: папки bot/ и human/ — CSV по UUID (боты и люди).
# Анализирует сырые логи, выделяет паттерны, предсказывает: бот | вероятно_бот | человек.
# ============================================================

import os
import json
import joblib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Config
BOT_DIR = "bot"
HUMAN_DIR = "human"
MODEL_PATH = "bot_detector_model.joblib"
FEATURES_PATH = "bot_detector_features.json"
VERDICT_PATH = "bot_verdict.csv"
VERDICT_HUMAN_THRESHOLD = 0.3
VERDICT_BOT_THRESHOLD = 0.6
NORMAL_UNIQUE_IPS = 3

REQUIRED_COLS = ["timestamp_shifted", "endpoint_code", "user_code", "ip_code"]

# Эндпоинты капчи (порядок: CHECK → ACTION)
CAPTCHA_CHECK_PREFIX = "CAPTCHA_CHECK"
CAPTCHA_ACTION_PREFIX = "CAPTCHA_ACTION"


# ============================================================
# LOAD & PREPARE
# ============================================================

def load_logs(path: str, chunksize: int | None = None) -> pd.DataFrame:
    """Загружает CSV с логами."""
    if chunksize:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunk = _prepare(chunk)
            if not chunk.empty:
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    df = pd.read_csv(path)
    return _prepare(df)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    df = df.dropna(subset=REQUIRED_COLS).copy()
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_shifted"
    df["timestamp"] = pd.to_datetime(df[ts_col], format="mixed", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["user_code", "timestamp"]).reset_index(drop=True)
    df["delta"] = df.groupby("user_code")["timestamp"].diff().dt.total_seconds()
    df = df[(df["delta"].isna()) | (df["delta"] >= 0)].copy()
    return df


# ============================================================
# FEATURE ENGINEERING (user-level)
# ============================================================

def _same_endpoint_ratio(group: pd.DataFrame) -> float:
    prev = group["endpoint_code"].shift(1)
    return float((group["endpoint_code"] == prev).mean())


def _run_stats(group: pd.DataFrame) -> tuple[float, float, dict]:
    block = (group["endpoint_code"] != group["endpoint_code"].shift(1)).cumsum()
    runs = group.groupby(block, sort=False).agg(
        endpoint_code=("endpoint_code", "first"),
        run_len=("endpoint_code", "size")
    ).reset_index(drop=True)
    if runs.empty:
        return 0.0, 0.0, {}
    mean_len = float(runs["run_len"].mean())
    max_len = int(runs["run_len"].max())
    by_ep = runs.groupby("endpoint_code")["run_len"].mean().sort_values(ascending=False).to_dict()
    return mean_len, max_len, by_ep


def _captcha_pass_speeds(group: pd.DataFrame) -> dict:
    """
    Скорость прохождения капчи: время (сек) между CAPTCHA_CHECK → CAPTCHA_ACTION.
    Считается ТОЛЬКО когда CAPTCHA_CHECK непосредственно предшествует CAPTCHA_ACTION
    (соседние строки отсортированные по timestamp).
    Если порядок другой — пара игнорируется.
    """
    endpoints = group["endpoint_code"].values
    timestamps = group["timestamp"].values
    speeds: list[float] = []

    for i in range(len(endpoints) - 1):
        ep_cur = str(endpoints[i])
        ep_nxt = str(endpoints[i + 1])
        if ep_cur.startswith(CAPTCHA_CHECK_PREFIX) and ep_nxt.startswith(CAPTCHA_ACTION_PREFIX):
            dt = (pd.Timestamp(timestamps[i + 1]) - pd.Timestamp(timestamps[i])).total_seconds()
            if dt >= 0:
                speeds.append(dt)

    if speeds:
        arr = np.array(speeds)
        return {
            "captcha_speed_count": len(arr),
            "captcha_speed_mean_sec": float(np.mean(arr)),
            "captcha_speed_median_sec": float(np.median(arr))
        }
    return {
        "captcha_speed_count": 0,
        "captcha_speed_mean_sec": 0.0,
        "captcha_speed_median_sec": 0.0,
    }


def build_user_features(group: pd.DataFrame) -> dict:
    """Строит числовые признаки по сессии пользователя."""
    group = group.sort_values("timestamp").copy()
    deltas = group["delta"].dropna()
    start = group["timestamp"].iloc[0]
    end = group["timestamp"].iloc[-1]
    duration_sec = max((end - start).total_seconds(), 1.0)
    n = len(group)

    per_sec = group.set_index("timestamp").resample("1s").size()
    per_min = group.set_index("timestamp").resample("1min").size()
    peak_sec = int(per_sec.max()) if not per_sec.empty else 0
    peak_min = int(per_min.max()) if not per_min.empty else 0

    ep_counts = group["endpoint_code"].value_counts()
    top_ep = ep_counts.index[0] if len(ep_counts) > 0 else ""
    top_count = int(ep_counts.iloc[0]) if len(ep_counts) > 0 else 0
    top_share = top_count / n if n > 0 else 0.0

    mean_run, max_run, by_ep = _run_stats(group)
    same_ratio = _same_endpoint_ratio(group)
    unique_ips = int(group["ip_code"].nunique())
    ip_switches = int((group["ip_code"] != group["ip_code"].shift(1)).sum() - 1)
    ip_switches = max(ip_switches, 0)

    # Endpoint counts (may not exist)
    def _cnt(name: str) -> int:
        return int(ep_counts.get(name, 0))

    timeslot = _cnt("TIMESLOT_AVAILABLE_DATES_001")
    res_search = _cnt("RESERVATION_SEARCH_001")
    res_card = _cnt("RESERVATION_CARD_001")
    captcha = _cnt("CAPTCHA_CHECK_001")
    captcha_action = _cnt("CAPTCHA_ACTION_001")
    polling = timeslot + res_search + res_card + captcha

    median_delta_ms = float(deltas.median() * 1000) if not deltas.empty else np.nan
    mean_delta_ms = float(deltas.mean() * 1000) if not deltas.empty else np.nan
    pct_10 = float((deltas < 0.01).mean()) if not deltas.empty else 0.0
    pct_50 = float((deltas < 0.05).mean()) if not deltas.empty else 0.0
    pause_5 = int((deltas > 5).sum()) if not deltas.empty else 0
    pause_10 = int((deltas > 10).sum()) if not deltas.empty else 0

    # Скорость прохождения капчи (CAPTCHA_CHECK → CAPTCHA_ACTION)
    captcha_speed = _captcha_pass_speeds(group)

    feats = {
        "total_requests": n,
        "duration_sec": duration_sec,
        "duration_hours": duration_sec / 3600.0,
        "avg_rps": n / duration_sec,
        "peak_req_per_sec": peak_sec,
        "peak_req_per_min": peak_min,
        "median_delta_ms": median_delta_ms if pd.notna(median_delta_ms) else 0.0,
        "mean_delta_ms": mean_delta_ms if pd.notna(mean_delta_ms) else 0.0,
        "pct_lt_10ms": pct_10,
        "pct_lt_50ms": pct_50,
        "pause_gt_5s": pause_5,
        "pause_gt_10s": pause_10,
        "unique_ips": unique_ips,
        "ip_switches": ip_switches,
        "ip_switch_rate": ip_switches / max(n - 1, 1),
        # Базовая норма: до 3 разных IP на пользователя.
        "ip_over_normal_3": max(unique_ips - NORMAL_UNIQUE_IPS, 0),
        "ip_many_flag": int(unique_ips > NORMAL_UNIQUE_IPS),
        "unique_endpoints": int(group["endpoint_code"].nunique()),
        "same_endpoint_ratio": same_ratio,
        "mean_run_len": mean_run,
        "max_run_len": max_run,
        "top_endpoint_count": top_count,
        "top_endpoint_share": top_share,
        "timeslot_count": timeslot,
        "reservation_search_count": res_search,
        "reservation_card_count": res_card,
        "captcha_count": captcha,
        "captcha_action_count": captcha_action,
        "polling_core_count": polling,
        "polling_core_share": polling / n if n > 0 else 0.0,
        "timeslot_mean_run_len": float(by_ep.get("TIMESLOT_AVAILABLE_DATES_001", 0.0)),
        "reservation_search_mean_run_len": float(by_ep.get("RESERVATION_SEARCH_001", 0.0)),
    }

    # Добавляем все признаки скорости прохождения капчи
    feats.update(captcha_speed)

    return feats


def aggregate_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует логи в один вектор признаков на user_code."""
    rows = []
    for user_code, group in df.groupby("user_code", sort=False):
        feats = build_user_features(group)
        feats["user_code"] = user_code
        rows.append(feats)
    return pd.DataFrame(rows)


# ============================================================
# SUPERVISOR: bot/ и human/
# ============================================================

def load_supervisor(bot_dir: str = BOT_DIR, human_dir: str = HUMAN_DIR) -> pd.DataFrame:
    """Загружает все CSV из bot/ и human/, объединяет с метками is_bot."""
    all_dfs = []
    for path, is_bot in [(bot_dir, 1), (human_dir, 0)]:
        folder = Path(path)
        if not folder.exists():
            continue
        for f in folder.glob("*.csv"):
            try:
                df = load_logs(str(f))
                if df.empty:
                    continue
                df["is_bot"] = is_bot
                all_dfs.append(df)
            except Exception as e:
                print(f"Пропуск {f}: {e}")
    if not all_dfs:
        raise FileNotFoundError(f"Нет CSV в {bot_dir}/ или {human_dir}/")
    return pd.concat(all_dfs, ignore_index=True)


# ============================================================
# TRAIN
# ============================================================

def train(
    bot_dir: str = BOT_DIR,
    human_dir: str = HUMAN_DIR,
    model_path: str = MODEL_PATH,
    features_path: str = FEATURES_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, list[str]]:
    """Обучает RandomForest на данных из bot/ и human/."""
    print("Загрузка супервизора (bot/ + human/)...")
    df = load_supervisor(bot_dir=bot_dir, human_dir=human_dir)
    print(f"Строк логов: {len(df):,}, пользователей: {df['user_code'].nunique()}")

    print("Построение признаков...")
    feat_df = aggregate_user_features(df)
    feat_df = feat_df.merge(
        df.groupby("user_code")["is_bot"].first().reset_index(),
        on="user_code",
        how="left"
    )
    feat_df = feat_df.dropna(subset=["is_bot"])
    feat_df["is_bot"] = feat_df["is_bot"].astype(int)

    exclude = ["user_code", "is_bot"]
    feature_cols = [c for c in feat_df.columns if c not in exclude]
    X = feat_df[feature_cols].fillna(0)
    y = feat_df["is_bot"]

    if y.nunique() < 2:
        raise ValueError("Нужны оба класса: добавьте CSV и в bot/, и в human/.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


    model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n=== Оценка модели ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["human", "bot"]))
    try:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    except Exception:
        pass

    joblib.dump(model, model_path)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nМодель сохранена: {model_path}")
    return model, feature_cols


# ============================================================
# PREDICT
# ============================================================

def predict(
    input_csv: str,
    model_path: str = MODEL_PATH,
    features_path: str = FEATURES_PATH,
    output_csv: str = VERDICT_PATH,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Предсказывает по сырым логам, сохраняет вердикт: бот | вероятно_бот | человек."""
    model = joblib.load(model_path)
    with open(features_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    print(f"Загрузка {input_csv} (чанками по {chunksize:,})...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunksize)):
        chunk = _prepare(chunk)
        if chunk.empty:
            continue
        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            print(f"  загружено {(i+1)*chunksize:,} строк...")
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if df.empty:
        raise ValueError("Нет данных в CSV.")

    print(f"Построение признаков для {df['user_code'].nunique()} пользователей...")
    feat_df = aggregate_user_features(df)

    X = feat_df[feature_cols].reindex(columns=feature_cols).fillna(0)
    proba = model.predict_proba(X)[:, 1]

    def verdict(p: float) -> str:
        if p < VERDICT_HUMAN_THRESHOLD:
            return "человек"
        if p < VERDICT_BOT_THRESHOLD:
            return "вероятно_бот"
        return "бот"

    result = pd.DataFrame({
        "user_code": feat_df["user_code"],
        "bot_probability": np.round(proba, 4),
        "verdict": [verdict(p) for p in proba],
        "total_requests": feat_df["total_requests"],
    })
    result.to_csv(output_csv, index=False)
    print(f"Вердикт сохранён: {output_csv} ({len(result)} пользователей)")
    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"], help="train или predict")
    parser.add_argument("--input", default="behavior_full_pured.csv", help="CSV для predict")
    parser.add_argument("--output", default=VERDICT_PATH, help="Файл вердикта")
    parser.add_argument("--bot-dir", default=BOT_DIR)
    parser.add_argument("--human-dir", default=HUMAN_DIR)
    args = parser.parse_args()

    if args.mode == "train":
        train(bot_dir=args.bot_dir, human_dir=args.human_dir)
    else:
        predict(input_csv=args.input, output_csv=args.output)