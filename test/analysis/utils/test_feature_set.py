"""Regression tests for FeatureSet property expressions.

These cover the operator-precedence bugs in ``regressors`` and ``all_columns``
where the original ``a + b or []`` form dropped ``a`` when ``b`` was ``None`` or
truncated the concatenation when any component was ``None``.
"""

from backtest.utils.feature_set import FeatureSet


def test_regressors_returns_numeric_features_when_categorical_none() -> None:
    fs = FeatureSet(numeric_features=["a", "b"], target="y", categorical_features=None)
    assert fs.regressors == ["a", "b"]


def test_regressors_concatenates_numeric_and_categorical_when_present() -> None:
    fs = FeatureSet(
        numeric_features=["a", "b"],
        target="y",
        categorical_features=["c1", "c2"],
    )
    assert fs.regressors == ["a", "b", "c1", "c2"]


def test_regressors_returns_numeric_when_categorical_empty_list() -> None:
    fs = FeatureSet(numeric_features=["a", "b"], target="y", categorical_features=[])
    assert fs.regressors == ["a", "b"]


def test_all_columns_concatenates_all_three_components() -> None:
    fs = FeatureSet(
        numeric_features=["a", "b"],
        target="y",
        categorical_features=["c1"],
        eval_fields=["e1", "e2"],
    )
    assert fs.all_columns == ["a", "b", "c1", "e1", "e2"]


def test_all_columns_falls_back_to_numeric_when_others_none() -> None:
    fs = FeatureSet(numeric_features=["a", "b"], target="y")
    # Regression: previous form ``num + cat + eval or []`` blew up on categorical=None.
    assert fs.all_columns == ["a", "b"]


def test_all_columns_handles_empty_lists() -> None:
    fs = FeatureSet(
        numeric_features=["a"],
        target="y",
        categorical_features=[],
        eval_fields=[],
    )
    assert fs.all_columns == ["a"]


def test_all_columns_with_only_eval_fields() -> None:
    fs = FeatureSet(
        numeric_features=[],
        target="y",
        categorical_features=None,
        eval_fields=["e1"],
    )
    assert fs.all_columns == ["e1"]


def test_auto_regressors_never_include_post_announcement_targets() -> None:
    regressors = FeatureSet.auto().regressors
    assert regressors
    assert not any(column.startswith("target_return@") for column in regressors)
