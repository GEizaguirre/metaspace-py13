import json
import re
import logging
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd

from sm.engine.errors import SMError
from sm.engine.storage import get_s3_client
from sm.engine.util import split_s3_path

logger = logging.getLogger('engine')


def add_derived_features(
    target_df: pd.DataFrame, decoy_df: pd.DataFrame, decoy_ratio: float, features: List[str]
):
    """Adds extra feature columns needed for the model to target_df and decoy_df.
    This is separate from the metric calculation in formula_validator as these derived features
    require statistics from a full ranking of targets & decoys, which isn't available in
    formula_validator .
    """
    # pylint: disable=import-outside-toplevel,cyclic-import  # circular import
    from sm.engine.annotation.fdr import score_to_fdr_map

    nonzero_targets = (target_df.chaos > 0) & (target_df.spatial > 0) & (target_df.spectral > 0)
    nonzero_decoys = (decoy_df.chaos > 0) & (decoy_df.spatial > 0) & (decoy_df.spectral > 0)

    fdr_features = [(f[: -len('_fdr')], f) for f in features if f.endswith('_fdr')]
    for feature, fdr_feature in fdr_features:
        target_values = target_df[feature].values
        decoy_values = decoy_df[feature].values
        if feature.startswith('mz_err'):
            # With mz_err features, 0 is the best value, and values get worse as they move away
            # from 0. They're transformed by the negative absolute value here so that
            # higher values are better. However, this transformed value is not interesting
            # to either users or debugging developers, so the temporary value is not stored
            # as a new feature.
            target_values = -np.abs(target_values)
            decoy_values = -np.abs(decoy_values)

        # Rule of Succession is disabled here because it would add an unnecessary bias at
        # by limiting the minimum value. It will eventually be applied in the final FDR ranking.
        fdr_map = score_to_fdr_map(
            target_values[nonzero_targets],
            decoy_values[nonzero_decoys],
            decoy_ratio,
            rule_of_succession=False,
            monotonic=True,
        )

        # fdr_map = fdr_map.clip(0.0, 1.0)
        target_df[fdr_feature] = np.where(
            nonzero_targets, fdr_map.reindex(target_values, fill_value=1.0).values, 1.0
        )
        decoy_df[fdr_feature] = np.where(
            nonzero_decoys, fdr_map.reindex(decoy_values, fill_value=1.0).values, 1.0
        )

    abserr_features = [(f[: -len('_abserr')], f) for f in features if f.endswith('_abserr')]
    for feature, abserr_feature in abserr_features:
        # With mz_err features, 0 is the best value, and values get worse as they move away
        # from 0 in either direction. They're transformed by the negative absolute value here
        # so that higher values are better. However, this transformed value is not interesting
        # to either users or debugging developers, so it's filtered out later.
        target_df[abserr_feature] = -target_df[feature].abs()
        decoy_df[abserr_feature] = -decoy_df[feature].abs()


def remove_uninteresting_features(target_df: pd.DataFrame, decoy_df: pd.DataFrame):
    uninteresting_features = [f for f in target_df.columns if f.endswith('_abserr')]
    target_df = target_df.drop(uninteresting_features, axis=1)
    decoy_df = decoy_df.drop(uninteresting_features, axis=1)
    return target_df, decoy_df


class ScoringModel:
    """Represents a scoring model to use as annotation base."""

    # pylint: disable=redefined-builtin
    def __init__(
        self,
        id: int = None,
        name: str = None,
        version: str = None,
        type: str = None,
        is_archived: bool = None,
    ):
        self.id = id
        self.name = name
        self.version = version
        self.type = type
        self.is_archived = is_archived

    def score(
        self, target_df: pd.DataFrame, decoy_df: pd.DataFrame, decoy_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Processes the targets & decoys from one FDR ranking and returns the dataframes with the
        'msm' column populated with the computed score, and potentially other columns added if they
        would help explain the score."""
        raise NotImplementedError()

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'is_archived': self.is_archived,
        }


def find_by_name_version(name: str, version: str) -> ScoringModel:
    # Import DB locally so that Lithops doesn't try to pickle it & fail due to psycopg2
    # pylint: disable=import-outside-toplevel  # circular import
    from sm.engine.db import DB

    data = DB().select_one_with_fields(
        'SELECT id, name, version, type FROM scoring_model WHERE name = %s AND version = %s',
        params=(name, version),
    )
    if not data:
        raise SMError(f'ScoringModel not found: {name}')
    return ScoringModel(**data)


def find_by_id(id_: int) -> ScoringModel:
    """Find scoring model by id."""
    # Import DB locally so that Lithops doesn't try to pickle it & fail due to psycopg2
    # pylint: disable=import-outside-toplevel  # circular import
    from sm.engine.db import DB

    data = DB().select_one_with_fields(
        'SELECT id, name, version, type FROM scoring_model WHERE id = %s', params=(id_,)
    )
    if not data:
        raise SMError(f'ScoringModel not found: {id_}')
    return ScoringModel(**data)


class MsmScoringModel(ScoringModel):
    def score(
        self, target_df: pd.DataFrame, decoy_df: pd.DataFrame, decoy_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # MSM column is already populated - just pass it through
        return target_df, decoy_df


def load_scoring_model(name: Optional[str], version: Optional[str] = None) -> ScoringModel:
    # Import DB locally so that Lithops doesn't try to pickle it & fail due to psycopg2
    # pylint: disable=import-outside-toplevel  # circular import
    from sm.engine.db import DB

    if name is None or version is None:
        return MsmScoringModel()

    row = DB().select_one(
        "SELECT type, params, id FROM scoring_model WHERE name = %s and version=%s",
        (
            name,
            version,
        ),
    )
    assert row, f'Scoring model {name} {version} not found'
    type_, params, id_ = row

    return MsmScoringModel()


def load_scoring_model_by_id(id_: Optional[int] = None) -> ScoringModel:
    # Import DB locally so that Lithops doesn't try to pickle it & fail due to psycopg2
    # pylint: disable=import-outside-toplevel  # circular import
    from sm.engine.db import DB

    if id_ is None:
        return MsmScoringModel()

    row = DB().select_one(
        "SELECT name, version, type, params, id FROM scoring_model WHERE id = %s",
        (id_,),
    )
    assert row, f'Scoring model {id_} not found'
    name, version, type_, params, id_ = row

    if type_ == 'catboost':
        bucket, key = split_s3_path(params['s3_path'])
        with TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / 'model.cbm'
            with model_file.open('wb') as f:
                f.write(get_s3_client().get_object(Bucket=bucket, Key=key)['Body'].read())
            model = CatBoost()
            model.load_model(str(model_file), 'cbm')

        return CatBoostScoringModel(name, model, params, id_, name, version)
    elif type_ == 'original':
        return MsmScoringModel()
    else:
        raise ValueError(f'Unsupported scoring model type: {type_}')


def save_scoring_model_to_db(name, type_, version, params, created_dt=None):
    """Adds/updates the scoring_model in the local database"""
    # Import DB locally so that Lithops doesn't try to pickle it & fail due to psycopg2
    # pylint: disable=import-outside-toplevel  # circular import
    from sm.engine.db import DB

    if not isinstance(params, str):
        params = json.dumps(params)

    if not created_dt:
        created_dt = datetime.utcnow()

    db = DB()
    if db.select_one(
        'SELECT * FROM scoring_model WHERE name = %s and version = %s',
        (
            name,
            version,
        ),
    ):
        logger.info(f'Updating existing scoring model {name}')
        DB().alter(
            'UPDATE scoring_model SET type = %s, version = %s, ' ' params = %s WHERE name = %s',
            (type_, version, params, name),
        )
    else:
        logger.info(f'Inserting new scoring model {name}')
        DB().alter(
            'INSERT INTO scoring_model(name, type, version, params, created_dt) '
            ' VALUES (%s, %s, %s, %s, %s)',
            (name, type_, version, params, created_dt),
        )
