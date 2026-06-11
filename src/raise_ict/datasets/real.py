"""Real dataset adapters for Tier-P RAISE-ICT experiments."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def _read_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing dataset file: {resolved}")
    if resolved.suffix.lower() == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved, low_memory=False)


def _read_csv_row_sample(path: str | Path, max_rows: int, seed: int, chunksize: int = 100_000) -> pd.DataFrame:
    resolved = Path(path)
    row_count = 0
    for chunk in pd.read_csv(resolved, usecols=[0], chunksize=chunksize, low_memory=False):
        row_count += len(chunk)
    if row_count <= max_rows:
        return pd.read_csv(resolved, low_memory=False)

    rng = pd.Series(range(row_count)).sample(n=max_rows, random_state=seed).sort_values().to_numpy()
    frames = []
    start = 0
    pointer = 0
    for chunk in pd.read_csv(resolved, chunksize=chunksize, low_memory=False):
        end = start + len(chunk)
        chunk_positions = []
        while pointer < len(rng) and rng[pointer] < end:
            if rng[pointer] >= start:
                chunk_positions.append(int(rng[pointer] - start))
            pointer += 1
        if chunk_positions:
            frames.append(chunk.iloc[chunk_positions])
        start = end
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_table_sample(path: str | Path, max_rows: int | None, seed: int) -> pd.DataFrame:
    resolved = Path(path)
    if max_rows is None or max_rows <= 0 or resolved.suffix.lower() != ".csv":
        return _read_table(resolved)
    return _read_csv_row_sample(resolved, int(max_rows), seed)


def _sample_split(frame: pd.DataFrame, max_rows: int | None, seed: int, label_column: str = "label") -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    if label_column in frame.columns and frame[label_column].nunique(dropna=True) > 1:
        sample, _ = train_test_split(
            frame,
            train_size=max_rows,
            random_state=seed,
            stratify=frame[label_column],
        )
        return sample.reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _standardize_common(frame: pd.DataFrame, dataset_id: str, split: str, attack_column: str | None) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(col).strip().lower().replace(" ", "_") for col in out.columns]
    if "label" not in out.columns:
        raise ValueError(f"{dataset_id} is missing a label column")
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
    if attack_column and attack_column in out.columns:
        out["attack_type"] = out[attack_column].astype(str)
    elif "attack_cat" in out.columns:
        out["attack_type"] = out["attack_cat"].astype(str)
    elif "type" in out.columns:
        out["attack_type"] = out["type"].astype(str)
    else:
        out["attack_type"] = out["label"].map({0: "benign"}).fillna("attack")
    out["dataset_id"] = dataset_id
    out["split"] = split
    out["group"] = split
    return out


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip().lower()).strip("_")
        for col in out.columns
    ]
    return out


def _apply_attack_type_holdout(frame: pd.DataFrame, dataset_config: Mapping[str, Any], seed: int) -> pd.DataFrame:
    """Create a binary IDS split with held-out attack families in the test set."""
    out = frame.copy()
    holdouts = {str(value).lower() for value in dataset_config.get("holdout_attack_types", [])}
    if not holdouts:
        raise ValueError("attack_type_holdout split requires holdout_attack_types")

    attack_type = out["attack_type"].astype(str).str.lower()
    benign = out["label"].astype(int) == 0
    heldout_attack = (~benign) & attack_type.isin(holdouts)
    if heldout_attack.sum() == 0:
        raise ValueError(f"No rows matched holdout_attack_types={sorted(holdouts)}")

    out["split"] = "train"
    out.loc[heldout_attack, "split"] = "test"
    benign_index = out[benign].index
    normal_test_size = float(dataset_config.get("holdout_normal_test_size", 0.3))
    if len(benign_index) > 1 and normal_test_size > 0.0:
        _, benign_test = train_test_split(
            benign_index,
            test_size=normal_test_size,
            random_state=seed,
        )
        out.loc[benign_test, "split"] = "test"
    out["group"] = out["attack_type"].astype(str)
    return out.reset_index(drop=True)


def load_unsw_nb15(dataset_config: Mapping[str, Any]) -> pd.DataFrame:
    """Load UNSW-NB15 train/test files and preserve the official split labels."""
    root = Path(dataset_config.get("data_root", "data/raw/unsw_nb15/temporal"))
    train_path = root / dataset_config.get("train_file", "train-00000-of-00001.parquet")
    test_path = root / dataset_config.get("test_file", "test-00000-of-00001.parquet")
    seed = int(dataset_config.get("seed", 0))
    train = _standardize_common(_read_table(train_path), "UNSW-NB15", "train", "attack_cat")
    test = _standardize_common(_read_table(test_path), "UNSW-NB15", "test", "attack_cat")
    train = _sample_split(train, dataset_config.get("max_train_rows"), seed)
    test = _sample_split(test, dataset_config.get("max_test_rows"), seed + 1)
    return pd.concat([train, test], ignore_index=True)


DEFAULT_CICIDS2017_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv.parquet",
    "Tuesday-WorkingHours.pcap_ISCX.csv.parquet",
    "Wednesday-workingHours.pcap_ISCX.csv.parquet",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
]

DEFAULT_CSE_CIC_IDS2018_FILES = [
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]


def _cic_day_and_scenario(filename: str) -> tuple[str, str]:
    stem = filename.replace(".pcap_ISCX.csv.parquet", "").replace(".parquet", "")
    day = stem.split("-", 1)[0].lower()
    return day, stem


def _cse_day_date_and_scenario(filename: str) -> tuple[str, str, str]:
    stem = filename.replace("_TrafficForML_CICFlowMeter.csv", "").replace(".csv", "").replace(".parquet", "")
    parts = stem.split("-")
    day = parts[0].lower()
    date = "-".join(parts[1:4]) if len(parts) >= 4 else stem
    return day, date, stem


def _standardize_cicids2017(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    out = _clean_columns(frame)
    if "label" not in out.columns:
        raise ValueError(f"CICIDS2017 file is missing label column: {filename}")
    day, scenario = _cic_day_and_scenario(filename)
    out["attack_type"] = out["label"].astype(str)
    out["label"] = (~out["attack_type"].str.lower().isin({"benign", "normal"})).astype(int)
    out["dataset_id"] = "CICIDS2017"
    out["day"] = day
    out["scenario"] = scenario
    out["group"] = scenario
    return out


def _standardize_cse_cic_ids2018(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    out = _clean_columns(frame)
    if "label" not in out.columns:
        raise ValueError(f"CSE-CIC-IDS2018 file is missing label column: {filename}")
    out = out[out["label"].astype(str).str.strip().str.lower() != "label"].copy()
    day, date, scenario = _cse_day_date_and_scenario(filename)
    out["attack_type"] = out["label"].astype(str)
    out["label"] = (~out["attack_type"].str.lower().isin({"benign", "normal"})).astype(int)
    out["dataset_id"] = "CSE-CIC-IDS2018"
    out["day"] = day
    out["date"] = date
    out["scenario"] = scenario
    out["group"] = scenario
    return out


def load_cicids2017(dataset_config: Mapping[str, Any]) -> pd.DataFrame:
    """Load CICIDS2017 flow files and apply day or scenario holdout splits."""
    root = Path(dataset_config.get("data_root", "data/raw/cicids2017/machine_learning"))
    files = dataset_config.get("files", DEFAULT_CICIDS2017_FILES)
    seed = int(dataset_config.get("seed", 0))
    max_rows_per_file = dataset_config.get("max_rows_per_file")
    frames = []
    for offset, filename in enumerate(files):
        frame = _standardize_cicids2017(_read_table(root / filename), filename)
        frame = _sample_split(frame, max_rows_per_file, seed + offset)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    split_strategy = dataset_config.get("split_strategy", "day_holdout")
    if split_strategy == "day_holdout":
        holdout_days = {str(day).lower() for day in dataset_config.get("holdout_days", ["friday"])}
        out["split"] = out["day"].astype(str).str.lower().isin(holdout_days).map({True: "test", False: "train"})
    elif split_strategy == "scenario_holdout":
        holdout_scenarios = {str(scenario) for scenario in dataset_config.get("holdout_scenarios", [])}
        if not holdout_scenarios:
            raise ValueError("scenario_holdout split requires holdout_scenarios")
        out["split"] = out["scenario"].astype(str).isin(holdout_scenarios).map({True: "test", False: "train"})
    return out.reset_index(drop=True)


def load_cse_cic_ids2018(dataset_config: Mapping[str, Any]) -> pd.DataFrame:
    """Load CSE-CIC-IDS2018 flow files and apply date/day/scenario splits."""
    root = Path(dataset_config.get("data_root", "data/raw/cse_cic_ids2018"))
    files = dataset_config.get("files", DEFAULT_CSE_CIC_IDS2018_FILES)
    seed = int(dataset_config.get("seed", 0))
    max_rows_per_file = dataset_config.get("max_rows_per_file")
    frames = []
    for offset, filename in enumerate(files):
        frame = _standardize_cse_cic_ids2018(
            _read_table_sample(root / filename, max_rows_per_file, seed + offset),
            filename,
        )
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    split_strategy = dataset_config.get("split_strategy", "date_holdout")
    if split_strategy == "date_holdout":
        holdout_dates = {str(date).lower() for date in dataset_config.get("holdout_dates", ["02-03-2018"])}
        out["split"] = out["date"].astype(str).str.lower().isin(holdout_dates).map({True: "test", False: "train"})
    elif split_strategy == "day_holdout":
        holdout_days = {str(day).lower() for day in dataset_config.get("holdout_days", ["friday"])}
        out["split"] = out["day"].astype(str).str.lower().isin(holdout_days).map({True: "test", False: "train"})
    elif split_strategy == "scenario_holdout":
        holdout_scenarios = {str(scenario) for scenario in dataset_config.get("holdout_scenarios", [])}
        if not holdout_scenarios:
            raise ValueError("scenario_holdout split requires holdout_scenarios")
        out["split"] = out["scenario"].astype(str).isin(holdout_scenarios).map({True: "test", False: "train"})
    return out.reset_index(drop=True)


def load_ton_iot_network(dataset_config: Mapping[str, Any]) -> pd.DataFrame:
    """Load TON_IoT network flows with optional attack-family holdout splitting."""
    path = Path(dataset_config.get("path", "data/raw/ton_iot/train_test_network.csv"))
    seed = int(dataset_config.get("seed", 0))
    frame = _standardize_common(_read_table(path), "TON_IoT", "full", "type")
    frame = _sample_split(frame, dataset_config.get("max_rows"), seed)
    if dataset_config.get("split_strategy") == "attack_type_holdout":
        frame = _apply_attack_type_holdout(frame, dataset_config, seed)
    if "group" in dataset_config:
        frame["group"] = str(dataset_config["group"])
    return frame.reset_index(drop=True)
