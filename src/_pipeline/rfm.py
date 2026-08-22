"""RFM calculation engine for the pipeline facade."""

from pathlib import Path

import numpy as np
import pandas as pd

from src._pipeline.artifacts import ArtifactStore
from src._pipeline.contracts import (
    RFM_BEST_SCORE,
    RFM_FREQUENCY_COLUMN,
    RFM_FREQUENCY_SCORE_COLUMN,
    RFM_HIGH_SCORE_MIN,
    RFM_LOW_SCORE_MAX,
    RFM_MONETARY_COLUMN,
    RFM_MONETARY_SCORE_COLUMN,
    RFM_OUTPUT_FILENAME,
    RFM_RECENCY_COLUMN,
    RFM_RECENCY_SCORE_COLUMN,
    RFM_REFERENCE_OFFSET_DAYS,
    RFM_REQUIRED_COLUMNS,
    RFM_SCORE_LABELS,
    RFM_SEGMENT_CHURNED,
    RFM_SEGMENT_COLUMN,
    RFM_SEGMENT_LOYAL,
    RFM_SEGMENT_NEW,
    RFM_SEGMENT_POTENTIAL,
    RFM_SEGMENT_VIP,
    STRING_DTYPE,
    STRICT_PARSING_ERRORS,
)
from src.runtime import RuntimeContext


RFM_PARTITIONS_DIRECTORY = "rfm_partitions"
RFM_PARTITION_FILENAME_TEMPLATE = "part_{index:02d}.csv"


class RFMEngine:
    """Compute or reuse hash-partitioned customer RFM artifacts."""

    def __init__(
        self,
        context: RuntimeContext,
        artifacts: ArtifactStore,
        chunksize: int,
    ) -> None:
        self.context = context
        self.artifacts = artifacts
        self.chunksize = chunksize
        self.runtime_rfm_path = context.aggregate_root / RFM_OUTPUT_FILENAME

    def calculate(
        self,
        transactions_path: Path,
        customer_col: str,
        date_col: str,
        amount_col: str,
        analysis_date: str | pd.Timestamp | None,
        partition_count: int,
        *,
        force: bool,
    ) -> tuple[pd.DataFrame, Path]:
        cached = self._load_cached_rfm(customer_col, force=force)
        if cached is not None:
            frame, path = cached
            return frame, path

        paths, max_date = self._partition_transactions(
            transactions_path,
            customer_col,
            date_col,
            amount_col,
            partition_count,
        )
        reference_date = self._resolve_rfm_reference_date(
            analysis_date,
            max_date,
        )
        rfm = self._aggregate_rfm_partitions(
            paths,
            customer_col,
            date_col,
            amount_col,
            reference_date,
        )
        rfm = self._score_rfm(rfm)
        rfm = self._assign_rfm_segments(rfm)
        rfm.to_csv(self.runtime_rfm_path, index=False)
        self.artifacts.record_status("RFM", "COMPUTED", self.runtime_rfm_path)
        return rfm, self.runtime_rfm_path

    def _load_cached_rfm(
        self,
        customer_col: str,
        *,
        force: bool,
    ) -> tuple[pd.DataFrame, Path] | None:
        if force:
            return None
        source = self.artifacts.find_reusable_csv(
            "RFM",
            self.runtime_rfm_path,
            RFM_OUTPUT_FILENAME,
            RFM_REQUIRED_COLUMNS,
        )
        if source is None:
            return None
        return pd.read_csv(source, dtype={customer_col: STRING_DTYPE}), source

    def _partition_transactions(
        self,
        transactions_path: Path,
        customer_col: str,
        date_col: str,
        amount_col: str,
        partition_count: int,
    ) -> tuple[list[Path], pd.Timestamp]:
        """Distribute parsed transactions into stable customer-hash CSV partitions."""
        partition_root = self.context.aggregate_root / RFM_PARTITIONS_DIRECTORY
        partition_root.mkdir(exist_ok=True)
        paths = [
            partition_root / RFM_PARTITION_FILENAME_TEMPLATE.format(index=index)
            for index in range(partition_count)
        ]
        for path in paths:
            if path.exists():
                path.unlink()
        max_date: pd.Timestamp | None = None
        initialized: set[int] = set()
        for chunk in pd.read_csv(
            transactions_path,
            usecols=[customer_col, date_col, amount_col],
            dtype={customer_col: STRING_DTYPE},
            chunksize=self.chunksize,
        ):
            chunk[date_col] = pd.to_datetime(
                chunk[date_col], errors=STRICT_PARSING_ERRORS
            )
            chunk[amount_col] = pd.to_numeric(
                chunk[amount_col], errors=STRICT_PARSING_ERRORS
            )
            chunk_max = pd.Timestamp(chunk[date_col].max())
            max_date = (
                chunk_max if max_date is None or chunk_max > max_date else max_date
            )
            buckets = (
                pd.util.hash_pandas_object(chunk[customer_col], index=False).to_numpy()
                % partition_count
            )
            for index in np.unique(buckets):
                chunk.loc[buckets == index].to_csv(
                    paths[int(index)],
                    mode="a",
                    header=int(index) not in initialized,
                    index=False,
                )
                initialized.add(int(index))
        if max_date is None:
            raise ValueError("Cannot calculate RFM from an empty transaction cache")
        return paths, max_date

    @staticmethod
    def _resolve_rfm_reference_date(
        analysis_date: str | pd.Timestamp | None,
        max_date: pd.Timestamp,
    ) -> pd.Timestamp:
        return (
            pd.Timestamp(analysis_date)
            if analysis_date is not None
            else pd.Timestamp(max_date.date())
            + pd.offsets.Day(RFM_REFERENCE_OFFSET_DAYS)
        )

    def _aggregate_rfm_partitions(
        self,
        paths: list[Path],
        customer_col: str,
        date_col: str,
        amount_col: str,
        reference_date: pd.Timestamp,
    ) -> pd.DataFrame:
        aggregates: list[pd.DataFrame] = []
        for path in paths:
            if path.is_file():
                partition = pd.read_csv(path, dtype={customer_col: STRING_DTYPE})
                partition[date_col] = pd.to_datetime(
                    partition[date_col], errors=STRICT_PARSING_ERRORS
                )
                aggregates.append(
                    self._aggregate_rfm_partition(
                        partition,
                        customer_col,
                        date_col,
                        amount_col,
                        reference_date,
                    )
                )
        return pd.concat(aggregates, ignore_index=True)

    @staticmethod
    def _aggregate_rfm_partition(
        frame: pd.DataFrame,
        customer_col: str,
        date_col: str,
        amount_col: str,
        reference_date: pd.Timestamp,
    ) -> pd.DataFrame:
        frame = frame.copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors=STRICT_PARSING_ERRORS)
        frame[amount_col] = pd.to_numeric(
            frame[amount_col], errors=STRICT_PARSING_ERRORS
        )
        return frame.groupby(customer_col, as_index=False).agg(
            **{
                RFM_RECENCY_COLUMN: (
                    date_col,
                    lambda dates: (reference_date - dates.max()).days,
                ),
                RFM_FREQUENCY_COLUMN: (date_col, "nunique"),
                RFM_MONETARY_COLUMN: (amount_col, "sum"),
            }
        )

    def _score_rfm(self, rfm: pd.DataFrame) -> pd.DataFrame:
        for value, ascending, score in (
            (RFM_RECENCY_COLUMN, False, RFM_RECENCY_SCORE_COLUMN),
            (RFM_FREQUENCY_COLUMN, True, RFM_FREQUENCY_SCORE_COLUMN),
            (RFM_MONETARY_COLUMN, True, RFM_MONETARY_SCORE_COLUMN),
        ):
            rfm[score] = self._score_rfm_metric(rfm[value], ascending=ascending)
        return rfm

    @staticmethod
    def _score_rfm_metric(values: pd.Series, *, ascending: bool) -> pd.Series:
        if len(values) == 1:
            return pd.Series(RFM_BEST_SCORE, index=values.index, dtype="int64")
        ranked = values.rank(method="average", pct=True, ascending=ascending)
        return pd.cut(
            ranked,
            bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=RFM_SCORE_LABELS,
            include_lowest=True,
        ).astype(int)

    def _assign_rfm_segments(self, rfm: pd.DataFrame) -> pd.DataFrame:
        rfm[RFM_SEGMENT_COLUMN] = [
            self.classify_segment(*scores)
            for scores in rfm[
                [
                    RFM_RECENCY_SCORE_COLUMN,
                    RFM_FREQUENCY_SCORE_COLUMN,
                    RFM_MONETARY_SCORE_COLUMN,
                ]
            ].itertuples(index=False, name=None)
        ]
        return rfm

    @staticmethod
    def classify_segment(r_score: int, f_score: int, m_score: int) -> str:
        if (r_score, f_score, m_score) == (
            RFM_BEST_SCORE,
            RFM_BEST_SCORE,
            RFM_BEST_SCORE,
        ):
            return RFM_SEGMENT_VIP
        if r_score >= RFM_HIGH_SCORE_MIN and f_score >= RFM_HIGH_SCORE_MIN:
            return RFM_SEGMENT_LOYAL
        if r_score == RFM_BEST_SCORE and f_score <= RFM_LOW_SCORE_MAX:
            return RFM_SEGMENT_NEW
        if r_score == RFM_SCORE_LABELS[0]:
            return RFM_SEGMENT_CHURNED
        return RFM_SEGMENT_POTENTIAL
