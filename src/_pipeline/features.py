"""Product image and text feature engineering for the pipeline facade."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import image as mpimg

from src._pipeline.artifacts import ArtifactStore
from src._pipeline.contracts import (
    IMAGE_FEATURE_REQUIRED_COLUMNS,
    IMAGE_MEAN_COLUMN,
    IMAGE_PATH_COLUMN,
    IMAGE_STD_COLUMN,
    PRODUCT_FEATURES_CACHE_FILENAME,
    PRODUCT_ID_COLUMN,
    PRODUCT_NAME_COLUMN,
    PRODUCT_NAME_LENGTH_COLUMN,
    STRING_DTYPE,
)
from src.runtime import RuntimeContext


IMAGE_RGB_CHANNEL_COUNT = 3


class ProductFeatureEngineer:
    """Build or reuse product-level text and image features."""

    def __init__(
        self,
        context: RuntimeContext,
        artifacts: ArtifactStore,
    ) -> None:
        self.context = context
        self.artifacts = artifacts
        self.runtime_product_features_path = (
            context.feature_root / PRODUCT_FEATURES_CACHE_FILENAME
        )

    def build(
        self,
        articles: pd.DataFrame,
        *,
        force: bool,
    ) -> tuple[pd.DataFrame, Path]:
        """Return reusable product features or calculate them from raw images."""
        source = None if force else self.artifacts.find_reusable_csv(
            "product features",
            self.runtime_product_features_path,
            PRODUCT_FEATURES_CACHE_FILENAME,
            IMAGE_FEATURE_REQUIRED_COLUMNS,
        )
        if source is not None:
            return (
                pd.read_csv(source, dtype={PRODUCT_ID_COLUMN: STRING_DTYPE}),
                source,
            )

        raw = self._require_raw_data_root()
        records = [
            self._extract_image_features(raw, product_id, product_name, image_path)
            for product_id, product_name, image_path in articles[
                [PRODUCT_ID_COLUMN, PRODUCT_NAME_COLUMN, IMAGE_PATH_COLUMN]
            ].itertuples(index=False)
        ]
        features = pd.DataFrame.from_records(records)
        features.to_csv(self.runtime_product_features_path, index=False)
        self.artifacts.record_status(
            "product features",
            "COMPUTED",
            self.runtime_product_features_path,
        )
        return features, self.runtime_product_features_path

    @staticmethod
    def _extract_image_features(
        raw_data_root: Path,
        product_id: object,
        product_name: object,
        image_path: str,
    ) -> dict[str, object]:
        """Calculate text and pixel features for one product."""
        record: dict[str, object] = {
            PRODUCT_ID_COLUMN: product_id,
            IMAGE_PATH_COLUMN: image_path,
            PRODUCT_NAME_LENGTH_COLUMN: (
                len(str(product_name)) if pd.notna(product_name) else np.nan
            ),
            IMAGE_MEAN_COLUMN: np.nan,
            IMAGE_STD_COLUMN: np.nan,
        }
        path = raw_data_root / image_path
        if not path.is_file():
            return record
        try:
            pixels = np.asarray(mpimg.imread(path))
            if pixels.ndim == 2:
                pixels = pixels[..., np.newaxis]
            pixels = pixels[..., :IMAGE_RGB_CHANNEL_COUNT]
            record[IMAGE_MEAN_COLUMN] = float(np.mean(pixels))
            record[IMAGE_STD_COLUMN] = float(np.std(pixels))
        except (OSError, ValueError, SyntaxError):
            pass
        return record

    def _require_raw_data_root(self) -> Path:
        if self.context.raw_data_root is None:
            raise ValueError(
                "A required full-data artifact was unavailable or invalid and no raw H&M dataset is attached. "
                "Attach the H&M competition input or set HM_RAW_DATA_DIR to rebuild it."
            )
        return self.context.raw_data_root
