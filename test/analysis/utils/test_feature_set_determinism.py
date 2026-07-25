"""Regression tests for :meth:`FeatureSet.auto` determinism.

The training path passes ``FeatureSet.auto().regressors`` as the column
selector when it builds the CatBoost ``Pool``. If ``auto()`` iterates a set
under the hood, the regressor list order becomes ``PYTHONHASHSEED``-dependent
between processes; two fresh Python processes then hand CatBoost identical
data in different column orders and its greedy tree splitter tie-breaks
differently. That produced ±0.05 magnitude nondeterminism in ``test_topkpauc``
between consecutive notebook re-runs (larger than the between-model gaps the
paper reports). These tests lock in an enum-declaration-order iteration.
"""

import os
import subprocess
import sys
import textwrap
from typing import List

from backtest.utils.feature_set import FeatureSet
from core.feature_type import FeatureType
from features.FeatureWriter import REGRESSOR_OFFSETS


def _expected_regressor_order() -> List[str]:
    """Regressor list order predicted by enum declaration order."""
    numeric: List[str] = []
    for feature_type in FeatureType:
        if feature_type == FeatureType.NUM_PREV_PUMP:
            continue
        numeric.extend(feature_type.col_names(offsets=REGRESSOR_OFFSETS))
    numeric.append(FeatureType.NUM_PREV_PUMP.lower())
    return numeric


def test_feature_set_auto_follows_enum_declaration_order() -> None:
    fs = FeatureSet.auto()
    assert fs.regressors == _expected_regressor_order()
    # A concrete anchor: ``asset_return@*`` comes first (declaration order in
    # :class:`FeatureType`), not any of the ``*_zscore@*`` variants.
    assert fs.regressors[0] == FeatureType.ASSET_RETURN.col_names(offsets=REGRESSOR_OFFSETS)[0]
    assert fs.regressors[-1] == FeatureType.NUM_PREV_PUMP.lower()


def test_feature_set_auto_is_stable_across_fresh_processes(tmp_path) -> None:
    """Two fresh Python processes with random ``PYTHONHASHSEED`` values must
    produce byte-identical regressor lists. Under the pre-fix ``set(list(...))``
    iteration this failed on every second invocation.
    """
    script = textwrap.dedent("""
        import json
        import sys
        sys.path.insert(0, {root!r})
        from backtest.utils.feature_set import FeatureSet
        print(json.dumps(FeatureSet.auto().regressors))
        """).format(root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    script_path = tmp_path / "dump_regressors.py"
    script_path.write_text(script)

    outputs: List[str] = []
    # Use several different PYTHONHASHSEED values to make sure the ordering
    # doesn't accidentally match by luck of the default seed.
    for seed in ("0", "1", "1000000", "random"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outputs.append(result.stdout.strip())

    # All runs must produce byte-identical regressor lists.
    assert len(set(outputs)) == 1, "FeatureSet.auto().regressors varies across PYTHONHASHSEED values: " + repr(
        set(outputs)
    )
