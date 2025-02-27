from api_integrated_llm.data_models.scorer_models import ConfusionMatrixMode
from api_integrated_llm.helpers.metrics_helper import (
    check_coverage,
    get_confision_matrix_list,
    get_confusion_matrix_cells,
)


def test_check_coverage_empty() -> None:
    is_non_zero_gold, is_covered = check_coverage(gold=[], pred=[])

    assert not is_non_zero_gold
    assert not is_covered


def test_check_coverage_no_gold() -> None:
    is_non_zero_gold, is_covered = check_coverage(gold=[], pred=["a"])

    assert not is_non_zero_gold
    assert not is_covered


def test_check_coverage_no_pred() -> None:
    is_non_zero_gold, is_covered = check_coverage(gold=["a"], pred=[])

    assert is_non_zero_gold
    assert not is_covered


def test_check_coverage_no_subset() -> None:
    is_non_zero_gold, is_covered = check_coverage(gold=["a"], pred=["b"])

    assert is_non_zero_gold
    assert not is_covered


def test_check_coverage_is_subset() -> None:
    is_non_zero_gold, is_covered = check_coverage(gold=["a"], pred=["a", "b"])

    assert is_non_zero_gold
    assert is_covered


def test_get_confusion_matrix_cells_set_empty() -> None:
    tp, fp, tn, fn = get_confusion_matrix_cells(
        gold=[],
        pred=[],
        mode=ConfusionMatrixMode.SET,
    )

    assert tp == 1
    assert fp == 0
    assert tn == 0
    assert fn == 0


def test_get_confusion_matrix_cells_set_non_empty() -> None:
    tp, fp, tn, fn = get_confusion_matrix_cells(
        gold=["a", "b", "a", "c", "d"],
        pred=["a", "b", "c", "d"],
        mode=ConfusionMatrixMode.SET,
    )

    assert tp == 4
    assert fp == 0
    assert tn == 0
    assert fn == 0


def test_get_confusion_matrix_cells_list_non_empty() -> None:
    tp, fp, tn, fn = get_confusion_matrix_cells(
        gold=["a", "b", "a", "c", "d"],
        pred=["a", "b", "c", "d"],
        mode=ConfusionMatrixMode.LIST,
    )

    assert tp == 2
    assert fp == 2
    assert tn == 0
    assert fn == 3


def test_get_confusion_matrix_cells_multiset_non_empty() -> None:
    tp, fp, tn, fn = get_confusion_matrix_cells(
        gold=["a", "b", "a", "c", "d"],
        pred=["a", "b", "c", "d"],
        mode=ConfusionMatrixMode.MULTISET,
    )

    assert tp == 4
    assert fp == 0
    assert tn == 0
    assert fn == 1


def test_get_confusion_matrix_cells_set_no_perfect_match_false_negative() -> None:
    tp, fp, tn, fn = get_confusion_matrix_cells(
        gold=["a", "b", "a", "c", "d"],
        pred=["a", "c", "d"],
        mode=ConfusionMatrixMode.SET,
    )

    assert tp == 3
    assert fp == 0
    assert tn == 0
    assert fn == 1


def test_get_confusion_matrix_cells_multiset_no_perfect_match_false_negative() -> None:
    tp, fp, tn, fn = get_confusion_matrix_cells(
        gold=["a", "b", "a", "c", "d"],
        pred=["a", "c", "d"],
        mode=ConfusionMatrixMode.MULTISET,
    )

    assert tp == 3
    assert fp == 0
    assert tn == 0
    assert fn == 2


def test_get_confusion_matrix_cells_list_no_perfect_match_false_negative() -> None:
    tp, fp, tn, fn = get_confusion_matrix_cells(
        gold=["a", "b", "a", "c", "d"],
        pred=["a", "c", "d"],
        mode=ConfusionMatrixMode.LIST,
    )

    assert tp == 1
    assert fp == 2
    assert tn == 0
    assert fn == 4


def test_get_confusion_matrix_cells_set_no_perfect_match_false_positive() -> None:
    tp, fp, tn, fn = get_confusion_matrix_cells(
        gold=["a", "b", "a", "c", "d"],
        pred=["a", "c", "d", "k"],
        mode=ConfusionMatrixMode.SET,
    )

    assert tp == 3
    assert fp == 1
    assert tn == 0
    assert fn == 1


def test_get_confusion_matrix_cells_multiset_no_perfect_match_false_positive() -> None:
    tp, fp, tn, fn = get_confusion_matrix_cells(
        gold=["a", "b", "a", "c", "d"],
        pred=["a", "c", "d", "k"],
        mode=ConfusionMatrixMode.SET,
    )

    assert tp == 3
    assert fp == 1
    assert tn == 0
    assert fn == 1


def test_get_confusion_matrix_cells_list_no_perfect_match_false_positive() -> None:
    tp, fp, tn, fn = get_confusion_matrix_cells(
        gold=["a", "b", "a", "c", "d"],
        pred=["a", "c", "d", "k"],
        mode=ConfusionMatrixMode.LIST,
    )

    assert tp == 1
    assert fp == 3
    assert tn == 0
    assert fn == 4


def test_get_confision_matrix_list() -> None:
    confusion_matrix_models, nonZeroGold, covered = get_confision_matrix_list(
        gold_answers=[["a", "b", "a", "c", "d"]],
        predicted_answers=[["a", "c", "d", "k"]],
        mode=ConfusionMatrixMode.LIST,
    )

    assert len(confusion_matrix_models) == 1
    assert nonZeroGold == 1
    assert covered == 0

    assert confusion_matrix_models[0].true_positive == 1
    assert confusion_matrix_models[0].false_positive == 3
    assert confusion_matrix_models[0].true_negative == 0
    assert confusion_matrix_models[0].false_negative == 4
