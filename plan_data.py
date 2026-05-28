"""Shared utilities for MVP fight plan generator.

Кеширует frame из ufc_decay_pipeline, чтобы style_clustering /
asymmetry_diagnostics / envelope не пересобирали 100 фич каждый раз.
Также извлекает per-fighter snapshot и предсказывает P(win).
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ufc_ablation_analysis import (
    DATA_PATH,
    build_all_history,
    load_clean_fights,
)
from ufc_decay_pipeline import build_decay_frame


ARTIFACT_DIR = Path("artifacts")
FRAME_CACHE = ARTIFACT_DIR / "_plan_decay_frame.parquet"
FRAME_META_CACHE = ARTIFACT_DIR / "_plan_decay_meta.joblib"
HGB_ARTIFACT = ARTIFACT_DIR / "ufc_decay_hist_gradient_boosting_sigmoid_calibrated.joblib"


def _per_side_profile_long(fights: pd.DataFrame) -> pd.DataFrame:
    """Длинная таблица: только колонки, которых нет в frame per-side
    (fighter_stance/weight_lbs/reach_cm уже привязаны через matchup-фичи)."""
    profile_cols = ["fighter_height_cm", "fighter_dob"]
    rows = []
    for side in (1, 2):
        cols = ["event_date", f"f_{side}_id"]
        cols += [f"f_{side}_{c}" for c in profile_cols]
        sub = fights[cols].copy()
        sub.columns = ["event_date", "fighter_id"] + profile_cols
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _per_side_layoff(history: pd.DataFrame) -> pd.DataFrame:
    return history[["fighter_id", "event_date", "days_since_last_fight"]].copy()


def _build_frame_fresh() -> tuple[
    pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame
]:
    """Возвращает frame + features + history + profile (per-side raw profile)."""
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    fights = load_clean_fights(DATA_PATH)
    frame, features, diff_features = build_decay_frame(fights)
    history = build_all_history(fights)
    profile = _per_side_profile_long(fights)
    return frame, features, diff_features, history, profile


HISTORY_CACHE = ARTIFACT_DIR / "_plan_decay_history.parquet"
PROFILE_CACHE = ARTIFACT_DIR / "_plan_decay_profile.parquet"


def load_frame(use_cache: bool = True) -> tuple[
    pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame
]:
    """Возвращает (frame, features, diff_features, history, profile). Кеш в parquet/joblib."""
    if (
        use_cache
        and FRAME_CACHE.exists()
        and FRAME_META_CACHE.exists()
        and HISTORY_CACHE.exists()
        and PROFILE_CACHE.exists()
    ):
        frame = pd.read_parquet(FRAME_CACHE)
        history = pd.read_parquet(HISTORY_CACHE)
        profile = pd.read_parquet(PROFILE_CACHE)
        meta = joblib.load(FRAME_META_CACHE)
        return frame, meta["features"], meta["diff_features"], history, profile

    frame, features, diff_features, history, profile = _build_frame_fresh()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    safe = frame.copy()
    for col in safe.columns:
        if safe[col].dtype == object:
            safe[col] = safe[col].astype("string")
    safe.to_parquet(FRAME_CACHE, index=False)
    history.to_parquet(HISTORY_CACHE, index=False)
    profile.to_parquet(PROFILE_CACHE, index=False)
    joblib.dump({"features": features, "diff_features": diff_features}, FRAME_META_CACHE)
    return frame, features, diff_features, history, profile


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    return joblib.load(HGB_ARTIFACT)


# ---------------------------------------------------------------------------
# Per-fighter latest snapshot
# ---------------------------------------------------------------------------


def _per_side_features(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("f_1_")
            and c not in {"f_1_name", "f_1_id"}]


POSITIONS = ["head", "body", "leg", "distance", "clinch", "ground"]

# Сила shrinkage в "псевдо-attempts"; ~30 — мягко, ~100 — жёстко.
SHRINKAGE_STRENGTH = 30.0


def _apply_position_shrinkage(latest: pd.DataFrame) -> pd.DataFrame:
    """Добавляет shrunken_<pos>_accuracy и shrunken_<pos>_defense, сдвинутые
    к UFC-mean пропорционально оценённому числу попыток.

    Формула: shrunken = (empirical × n + prior × k) / (n + k),
    где n ≈ attempted_per_min × career_minutes, k = SHRINKAGE_STRENGTH.
    """
    duration = latest["avg_fight_duration_min"].fillna(0)
    n_fights = latest["num_career_fights"].fillna(0).astype(float)
    career_min = duration * n_fights

    # UFC-wide priors: вес по числу attempts (более стабильные позиции
    # получают точнее prior).
    for pos in POSITIONS:
        acc_col = f"avg_{pos}_accuracy"
        def_col = f"avg_{pos}_defense"
        vol_col = f"avg_{pos}_attempted_per_min"
        if acc_col not in latest.columns:
            continue
        prior_acc = latest[acc_col].median(skipna=True)
        prior_def = latest[def_col].median(skipna=True)
        n_att = (latest[vol_col].fillna(0) * career_min).clip(lower=0)
        k = SHRINKAGE_STRENGTH

        emp_acc = latest[acc_col]
        emp_def = latest[def_col]

        latest[f"shrunken_{pos}_accuracy"] = (
            emp_acc.fillna(prior_acc) * n_att + prior_acc * k
        ) / (n_att + k)
        latest[f"shrunken_{pos}_defense"] = (
            emp_def.fillna(prior_def) * n_att + prior_def * k
        ) / (n_att + k)
        # сохраняем эффективный n для inspection
        latest[f"effective_n_{pos}_attempts"] = n_att
    return latest


def build_fighter_index(
    frame: pd.DataFrame, history: pd.DataFrame, profile: pd.DataFrame
) -> pd.DataFrame:
    """Возвращает per-fighter latest snapshot.

    Каждая строка — один боец (по f_*_id). Колонки:
      fighter_id, fighter_name, event_date (последний бой), num_career_fights,
      все per-side метрики из frame (без префикса) +
      все cumulative/last-fight метрики из history.
    """
    side_metric_cols = [
        c[len("f_1_"):] for c in frame.columns
        if c.startswith("f_1_") and c not in {"f_1_name", "f_1_id"}
    ]

    rows = []
    for side in (1, 2):
        cols = ["event_date", f"f_{side}_id", f"f_{side}_name"]
        cols += [f"f_{side}_{m}" for m in side_metric_cols]
        sub = frame[cols].copy()
        sub.columns = ["event_date", "fighter_id", "fighter_name"] + side_metric_cols
        rows.append(sub)

    long = pd.concat(rows, ignore_index=True)
    long = long.dropna(subset=["fighter_id"])
    long = long.sort_values(["fighter_id", "event_date"])

    # history содержит recent_form/ewma колонки, пересекающиеся с per-side
    # frame EWMA — дропаем их (модель использует только frame-овые halflife=5).
    drop_overlap = [
        c for c in history.columns
        if c.startswith("recent_") or c.startswith("ewma_")
    ]
    history_trim = history.drop(columns=drop_overlap)
    long = long.merge(history_trim, on=["fighter_id", "event_date"], how="left")
    long = long.merge(profile, on=["fighter_id", "event_date"], how="left")

    dob = pd.to_datetime(long["fighter_dob"], errors="coerce")
    long["age_years"] = (long["event_date"] - dob).dt.days / 365.25
    long["height_cm"] = pd.to_numeric(long["fighter_height_cm"], errors="coerce")
    long["reach_cm"] = pd.to_numeric(long["fighter_reach_cm"], errors="coerce")
    long["weight_lbs"] = pd.to_numeric(long["fighter_weight_lbs"], errors="coerce")
    long["southpaw"] = (
        long["fighter_stance"].fillna("").astype(str).str.lower().eq("southpaw").astype(int)
    )

    latest = long.groupby("fighter_id", as_index=False).tail(1).reset_index(drop=True)
    fight_counts = long.groupby("fighter_id").size().rename("num_career_fights")
    latest = latest.merge(fight_counts, on="fighter_id", how="left")
    return latest


def find_fighter(index: pd.DataFrame, name_query: str) -> pd.Series:
    """Возвращает строку из fighter_index по подстроке/полному имени."""
    q = name_query.strip().lower()
    matches = index[index["fighter_name"].str.lower().str.contains(q, na=False, regex=False)]
    if len(matches) == 0:
        raise KeyError(f"Бойца '{name_query}' не нашёл")
    if len(matches) > 1:
        # вернуть самого свежего
        matches = matches.sort_values("event_date").tail(1)
    return matches.iloc[0]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


# aliases для случаев, где имя _diff не совпадает с базовой колонкой snapshot'а
DIFF_BASE_ALIAS = {
    "weight_cut": "weight_cut_lbs",
}


def _resolve_snap_value(snap: pd.Series, base: str) -> Any:
    if base in snap.index:
        return snap[base]
    alias = DIFF_BASE_ALIAS.get(base)
    if alias and alias in snap.index:
        return snap[alias]
    return np.nan


def build_pair_features(
    snap_a: pd.Series, snap_b: pd.Series, features: list[str], medians: dict[str, float]
) -> pd.DataFrame:
    """Собирает строку с 100 фичами из снапшотов двух бойцов.

    Антисимметричные _diff = a − b. Симметричные берутся из snapshot, либо
    вычисляются (layoff_*, weight_cut_*) либо падают на median.
    """
    row: dict[str, float] = {}
    layoff_a = snap_a.get("layoff_days", np.nan)
    layoff_b = snap_b.get("layoff_days", np.nan)
    cut_a = snap_a.get("weight_cut_lbs", np.nan)
    cut_b = snap_b.get("weight_cut_lbs", np.nan)

    for feat in features:
        val: float | None = None
        if feat.endswith("_diff"):
            base = feat[:-len("_diff")]
            va = _resolve_snap_value(snap_a, base)
            vb = _resolve_snap_value(snap_b, base)
            if not (pd.isna(va) or pd.isna(vb)):
                val = float(va) - float(vb)
            elif feat == "layoff_abs_diff" and not (pd.isna(layoff_a) or pd.isna(layoff_b)):
                val = abs(float(layoff_a) - float(layoff_b))
        else:
            # симметричные spec-cases
            if feat == "layoff_max" and not (pd.isna(layoff_a) or pd.isna(layoff_b)):
                val = float(max(layoff_a, layoff_b))
            elif feat == "layoff_min" and not (pd.isna(layoff_a) or pd.isna(layoff_b)):
                val = float(min(layoff_a, layoff_b))
            elif feat == "weight_cut_max" and not (pd.isna(cut_a) or pd.isna(cut_b)):
                val = float(max(cut_a, cut_b))
            elif feat == "weight_cut_sum" and not (pd.isna(cut_a) or pd.isna(cut_b)):
                val = float(cut_a + cut_b)
            elif feat == "stance_mismatch":
                sa = snap_a.get("southpaw", 0) or 0
                sb = snap_b.get("southpaw", 0) or 0
                val = int(sa != sb)
            elif feat == "same_stance":
                sa = str(snap_a.get("fighter_stance", "")).lower()
                sb = str(snap_b.get("fighter_stance", "")).lower()
                val = int(sa == sb and sa != "")
            elif feat == "style_clash":
                lean_a = snap_a.get("style_lean", np.nan)
                lean_b = snap_b.get("style_lean", np.nan)
                if not (pd.isna(lean_a) or pd.isna(lean_b)):
                    val = float(lean_a) * float(lean_b)
            else:
                v = snap_a.get(feat, np.nan)
                if not pd.isna(v):
                    val = float(v)

        if val is None:
            val = medians.get(feat, np.nan)
            val = float(val) if not pd.isna(val) else np.nan
        row[feat] = val

    return pd.DataFrame([row], columns=features)


def predict_p_win(snap_a: pd.Series, snap_b: pd.Series) -> dict[str, Any]:
    art = load_model()
    features = art["feature_columns"]
    medians = art["feature_medians"]
    model = art["model"]

    X = build_pair_features(snap_a, snap_b, features, medians)
    p = float(model.predict_proba(X)[0, 1])
    n_missing = int(X.isna().sum(axis=1).iloc[0])
    return {
        "p_a_wins": p,
        "n_features_missing": n_missing,
        "n_features_total": len(features),
    }
