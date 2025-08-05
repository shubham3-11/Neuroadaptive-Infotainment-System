"""
SleepinessDataset class for AdVitam Experiment 4 sleepiness detection data.

This module provides a unified interface for loading and preprocessing the AdVitam
Experiment 4 dataset for sleepiness detection tasks using physiological signals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Literal, Dict, Any
import warnings

# --- Hard-coded base dataset directory ------------------------------------
# Determine project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Use absolute data path so imports work from any cwd
DATA_ROOT = PROJECT_ROOT / "data"


class SleepinessDataset:
    """Dataset wrapper for AdVitam Exp4 physiological features + KSS labels.

    Parameters
    ----------
    segment : int, default 5
        How many equal-length segments to split each 30-min drive period into.
        Valid values: 1, 2, 5, 10, 20, 30.
    variant : {'periods'}, default 'periods'
        Which pre-segmented physio file to load (at the moment only the 5-min
        *period* segmentation is available).
    output_format : {'pandas'}, default 'pandas'
        Returned object type.
    data_root : str | Path, optional
        Override location of the dataset (defaults to project/.data/AdVitam/Exp4).
    feature_type : {'delta','driving','baseline','all'}, default 'delta'
        Which physio columns to keep:
        * delta    → Dr-Bl only
        * driving  → Dr (raw driving)
        * baseline → Bl (baseline before drive)
        * all      → every physio column
    """

    def __init__(
        self,
        segment: int = 5,
        variant: Literal["periods"] = "periods",
        output_format: Literal["pandas"] = "pandas",
        data_root: Optional[str | Path] = None,
        feature_type: Literal["delta", "driving", "baseline", "all"] = "delta",
    ) -> None:
        self._validate_parameters(segment, variant, output_format, feature_type)
        self.segment = segment
        self.variant = variant
        self.output_format = output_format
        self.feature_type = feature_type

        # --- resolve paths --------------------------------------------------
        # Default root now points to data/AdVitam (not Exp4-specific)
        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else DATA_ROOT / "AdVitam"
        )

        # e.g. data/AdVitam/preprocessed/physio/periods/features_segm_5.csv
        self.physio_path = (
            self.data_root
            / "preprocessed"
            / "physio"
            / variant
            / f"features_segm_{segment}.csv"
        )

        # e.g. data/AdVitam/preprocessed/questionnaire/Exp4_Database.csv
        self.questionnaire_path = (
            self.data_root
            / "preprocessed"
            / "questionnaire"
            / "Exp4_Database.csv"
        )

        # will be populated on load()
        self.data: Optional[pd.DataFrame] = None
        self.feature_cols: list[str] = []
        self.metadata_cols = [
            "subject_id",
            "label_sleep",
            "first_scenario",
            "label_time_exp",
            "period",
            "segment_id",
            "kss",
        ]
        self.target_col = "kss"

    # ---------------------------------------------------------------------
    # validation helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _validate_parameters(segment, variant, output_format, feature_type):
        if segment not in [1, 2, 5, 10, 20, 30]:
            raise ValueError("segment must be one of 1,2,5,10,20,30")
        if variant != "periods":
            raise ValueError("Only 'periods' variant supported for now")
        if output_format != "pandas":
            raise ValueError("Only pandas output supported")
        if feature_type not in {"delta", "driving", "baseline", "all"}:
            raise ValueError("Unknown feature_type")

    # ------------------------------------------------------------------
    # core loaders
    # ------------------------------------------------------------------

    def _load_physio(self) -> pd.DataFrame:
        if not self.physio_path.exists():
            raise FileNotFoundError(self.physio_path)
        df = pd.read_csv(self.physio_path)

        if self.feature_type == "delta":
            self.feature_cols = [c for c in df.columns if c.endswith("_Dr-Bl")]
        elif self.feature_type == "driving":
            self.feature_cols = [c for c in df.columns if c.endswith("_Dr") and not c.endswith("_Dr-Bl")]
        elif self.feature_type == "baseline":
            self.feature_cols = [c for c in df.columns if c.endswith("_Bl")]
        else:
            self.feature_cols = [
                c for c in df.columns if any(c.endswith(s) for s in ("_Bl", "_Dr", "_Dr-Bl"))
            ]

        cols = [
            "subject_id",
            "label_sleep",
            "label_first_scenario",
            "label_time_exp",
            "period",
            "segment_id",
        ] + self.feature_cols
        return df[cols].rename(columns={"label_first_scenario": "first_scenario"})

    def _load_questionnaire(self) -> pd.DataFrame:
        if not self.questionnaire_path.exists():
            raise FileNotFoundError(self.questionnaire_path)
        db = pd.read_csv(self.questionnaire_path)
        return self._interpolate_kss(db)

    # ---------------------------------------------------------------
    # KSS interpolation helpers
    # ---------------------------------------------------------------

    def _interpolate_kss(self, db: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, r in db.iterrows():
            self._process_participant_kss(r, rows, method)
        return pd.DataFrame(rows)

    def _process_participant_kss(self, r: pd.Series, out: list[dict], method: str):
        subj = int(r["participant_code"].split("_")[0])
        order = r["order_scenario"]
        scenarios = (
            [
                {"period": "Rural", "b": r["KSS_B_1"], "e": r["KSS_1"]},
                {"period": "Urban", "b": r["KSS_B_2"], "e": r["KSS_2"]},
            ]
            if order == 1
            else [
                {"period": "Urban", "b": r["KSS_B_1"], "e": r["KSS_1"]},
                {"period": "Rural", "b": r["KSS_B_2"], "e": r["KSS_2"]},
            ]
        )
        first_scenario = scenarios[0]["period"]
        for sc in scenarios:
            if pd.isna(sc["b"]) or pd.isna(sc["e"]):
                continue
            kss_vals = self._gen_kss(sc["b"], sc["e"], self.segment, method)
            for sid, val in enumerate(kss_vals):
                out.append(
                    {
                        "subject_id": subj,
                        "first_scenario": first_scenario,
                        "period": sc["period"],
                        "segment_id": sid,
                        "kss": val,
                    }
                )

    @staticmethod
    def _gen_kss(kss_b: float, kss_e: float, n: int, method: str) -> list[float]:
        if n == 1:
            return [kss_e]
        if method == "linear":
            return [kss_b + (kss_e - kss_b) * (i / (n - 1)) for i in range(n)]
        if method == "step":
            return [kss_b] * (n - 1) + [kss_e]
        raise ValueError("method must be 'linear' or 'step'")

    # ---------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------

    def load(
        self, *, clean: bool = True, missing_threshold: float = 0.5, kss_method: str = "linear"
    ) -> pd.DataFrame:
        """Load physio + KSS into one DataFrame."""
        physio = self._load_physio()
        kss = self._load_questionnaire()
        df = physio.merge(kss, on=["subject_id", "first_scenario", "period", "segment_id"], how="left")
        # drop rows without target
        df = df.dropna(subset=["kss"]).reset_index(drop=True)

        if clean:
            df = self._clean(df, missing_threshold)
        self.data = df
        return df

    def _clean(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        # simple cleaning: drop feature columns with too many NaNs
        nan_rate = df[self.feature_cols].isna().mean()
        drop_cols = nan_rate[nan_rate > threshold].index.tolist()
        if drop_cols:
            df = df.drop(columns=drop_cols)
            self.feature_cols = [c for c in self.feature_cols if c not in drop_cols]
        # do NOT drop rows—missing values handled downstream
        return df

    # Convenience getters
    def get_features(self) -> pd.DataFrame:
        if self.data is None:
            raise RuntimeError("call load() first")
        return self.data[self.feature_cols]

    def get_target(self) -> pd.Series:
        if self.data is None:
            raise RuntimeError("call load() first")
        return self.data[self.target_col]

    def get_metadata(self) -> pd.DataFrame:
        if self.data is None:
            raise RuntimeError("call load() first")
        return self.data[[c for c in self.metadata_cols if c in self.data.columns]]


# ------------------------------------------------------------------
# utility shortcut
# ------------------------------------------------------------------

def load_sleepiness_data(segment: int = 5, **kwargs) -> pd.DataFrame:
    """One-liner helper to get a prepared dataset."""
    return SleepinessDataset(segment=segment, **kwargs).load() 