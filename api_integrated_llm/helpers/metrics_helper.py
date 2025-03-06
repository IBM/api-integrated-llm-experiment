from collections import Counter
from typing import Dict, List, Tuple


from api_integrated_llm.data_models.scorer_models import (
    ConfusionMatrixMode,
    ConfusionMatrixModel,
)


def check_coverage(gold: List[str], pred: List[str]) -> Tuple[bool, bool]:
    is_non_zero_gold = False
    is_covered = False
    if len(gold) > 0:
        is_non_zero_gold = True
        gold_set = set(gold)
        pred_set = set(pred)
        if gold_set.issubset(pred_set):
            is_covered = True

    return is_non_zero_gold, is_covered


def get_confusion_matrix_cells(
    gold: List[str],
    pred: List[str],
    mode: ConfusionMatrixMode,
) -> Tuple[int, int, int, int]:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    if len(gold) == 0 and len(pred) == 0:
        true_positive = 1
        false_positive = 0
        true_negative = 0
        false_negative = 0
    else:
        if mode == ConfusionMatrixMode.SET or mode == ConfusionMatrixMode.COUNTER:
            gold_dict = Counter(gold)
            pred_dict = Counter(pred)

            for key, frequency in gold_dict.items():
                if key in pred_dict:
                    if mode == ConfusionMatrixMode.SET:
                        true_positive += 1
                    else:  # mode == ConfusionMatrixMode.COUNTER:
                        true_positive += min(frequency, pred_dict[key])
                        false_negative += max(0, frequency - pred_dict[key])
                        false_positive += max(0, pred_dict[key] - frequency)

                    pred_dict.pop(key)  # remove visited key in prediction dictionary
                else:  # key not in pred_dict
                    false_negative += (
                        1 if mode == ConfusionMatrixMode.SET else frequency
                    )

            for (
                key,
                frequency,
            ) in (
                pred_dict.items()
            ):  # handle all keys in prediction not existing in gold
                false_positive += 1 if mode == ConfusionMatrixMode.SET else frequency
        elif mode == ConfusionMatrixMode.LIST:  # list mode: sequence-aware approach
            num_matches = 0
            for idx, value in enumerate(gold):
                if len(pred) > idx:
                    if value == pred[idx]:
                        num_matches += 1
                    else:
                        break
                else:
                    break
            true_positive += num_matches
            false_positive += max(0, len(pred) - num_matches)
            false_negative += max(0, len(gold) - num_matches)
        else:
            raise Exception(
                "Undefined confusion matrix mode at get_confusion_matrix_cells()"
            )

    return (true_positive, false_positive, true_negative, false_negative)


def get_confision_matrix_from_answers(
    gold_answers: List[List[str]],
    predicted_answers: List[List[str]],
    mode: ConfusionMatrixMode = ConfusionMatrixMode.SET,
) -> ConfusionMatrixModel:
    matrix = ConfusionMatrixModel(mode=mode)

    for gold, pred in zip(gold_answers, predicted_answers):
        is_non_zero_gold, is_covered = check_coverage(gold=gold, pred=pred)
        matrix.num_non_zero_gold += 1 if is_non_zero_gold else 0
        matrix.num_is_covered += 1 if is_covered else 0
        (
            true_positive,
            false_positive,
            true_negative,
            false_negative,
        ) = get_confusion_matrix_cells(
            gold=gold,
            pred=pred,
            mode=mode,
        )

        matrix.true_positive += true_positive
        matrix.true_negative += true_negative
        matrix.false_positive += false_positive
        matrix.false_negative += false_negative

    return matrix


def get_confision_matrix_from_answers_by_output_length(
    gold_answers: List[List[str]],
    predicted_answers: List[List[str]],
    mode: ConfusionMatrixMode = ConfusionMatrixMode.SET,
) -> Dict[int, ConfusionMatrixModel]:
    output: Dict[int, ConfusionMatrixModel] = dict()

    for gold, pred in zip(gold_answers, predicted_answers):
        gold_length = len(gold)
        if gold_length not in output:
            output[gold_length] = ConfusionMatrixModel(mode=mode)
        is_non_zero_gold, is_covered = check_coverage(gold=gold, pred=pred)
        output[gold_length].num_non_zero_gold += 1 if is_non_zero_gold else 0
        output[gold_length].num_is_covered += 1 if is_covered else 0
        (
            true_positive,
            false_positive,
            true_negative,
            false_negative,
        ) = get_confusion_matrix_cells(
            gold=gold,
            pred=pred,
            mode=mode,
        )

        output[gold_length].true_positive += true_positive
        output[gold_length].true_negative += true_negative
        output[gold_length].false_positive += false_positive
        output[gold_length].false_negative += false_negative

    return output


# def _compute_confusion_mat(gold_answers: list, predicted_answers: list):
#     """
#     Returns number of True Positive, False Positive, False Negative per question, and coverage stats. Private function
#     """

#     num_TP = []
#     num_FP = []
#     num_FN = []
#     nonZeroGold = 0
#     covered = 0

#     for gold, pred in zip(gold_answers, predicted_answers):
#         tp, fp, fn = 0, 0, 0
#         for e in pred:
#             if e in gold:
#                 tp += 1
#             else:
#                 fp += 1
#         for e in gold:
#             if e not in pred:
#                 fn += 1

#         if len(gold) == 0:
#             if len(pred) == 0:
#                 tp, fp, fn = 1, 0, 0
#             else:
#                 tp, fp, fn = -1, 0, 0
#         else:
#             nonZeroGold += 1
#             if gold.issubset(pred):
#                 covered += 1

#         if len(pred) == 0:
#             if len(gold) != 0:
#                 tp, fp, fn = -2, 0, 0

#         num_TP.append(tp)
#         num_FP.append(fp)
#         num_FN.append(fn)

#     return num_TP, num_FP, num_FN, nonZeroGold, covered


# def _compute_metrics(
#     num_TP: list, num_FP: list, num_FN: list, nonZeroGold: int, covered: int
# ):
#     """
#     Computes [micro, macro] x [Precision, Recall, F1], Private function
#     """

#     prec_micro, recall_micro, f1_micro = 0.0, 0.0, 0.0
#     prec_macro, recall_macro, f1_macro = 0.0, 0.0, 0.0

#     tp, fp, fn = [np.array(x).astype(float) for x in (num_TP, num_FP, num_FN)]

#     prec_macro = np.nan_to_num(tp / (tp + fp))
#     recall_macro = np.nan_to_num(tp / (tp + fn))
#     idxs_special_case1 = np.where(tp == -1)[0]
#     idxs_special_case2 = np.where(tp == -2)[0]
#     prec_macro[idxs_special_case1] = 0.0  # type: ignore
#     recall_macro[idxs_special_case1] = 1.0  # type: ignore
#     prec_macro[idxs_special_case2] = 1.0  # type: ignore
#     recall_macro[idxs_special_case2] = 0.0  # type: ignore
#     f1_macro = np.nan_to_num(
#         2 * prec_macro * recall_macro / (prec_macro + recall_macro)
#     )
#     tp[idxs_special_case1] = 0
#     tp[idxs_special_case2] = 0

#     prec_micro = np.nan_to_num(tp.sum() / (tp.sum() + fp.sum()))
#     recall_micro = np.nan_to_num(tp.sum() / (tp.sum() + fn.sum()))
#     f1_micro = np.nan_to_num(
#         2 * prec_micro * recall_micro / (prec_micro + recall_micro)
#     )

#     all_metrics = {
#         "micro": {"precision": prec_micro, "recall": recall_micro, "f1": f1_micro},
#         "macro": {
#             "precision": {
#                 "score": prec_macro.mean(),
#             },
#             "recall": {"score": recall_macro.mean()},
#             "f1": {"score": f1_macro.mean()},
#             # 'f1-QALD': f1_macro_qald
#         },
#         "Non-empty gold": nonZeroGold,
#         "Covered": covered,
#     }

#     return all_metrics
