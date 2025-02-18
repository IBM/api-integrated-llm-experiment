import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

binarizer = MultiLabelBinarizer()


def compute_score_sklearn(gold_output, pred_output):
    binarizer.fit(gold_output)

    f1_score_macro = f1_score(
        binarizer.transform(gold_output),
        binarizer.transform(pred_output),
        average="macro",
    )
    precision_macro = precision_score(
        binarizer.transform(gold_output),
        binarizer.transform(pred_output),
        average="macro",
    )
    recall_macro = recall_score(
        binarizer.transform(gold_output),
        binarizer.transform(pred_output),
        average="macro",
    )

    return precision_macro, recall_macro, f1_score_macro


def _compute_confusion_mat(gold_answers: list, predicted_answers: list):
    """
    Returns number of True Positive, False Positive, False Negative per question, and coverage stats. Private function
    """

    num_TP = []
    num_FP = []
    num_FN = []
    nonZeroGold = 0
    covered = 0

    for gold, pred in zip(gold_answers, predicted_answers):
        tp, fp, fn = 0, 0, 0
        for e in pred:
            if e in gold:
                tp += 1
            else:
                fp += 1
        for e in gold:
            if e not in pred:
                fn += 1

        if len(gold) == 0:
            if len(pred) == 0:
                tp, fp, fn = 1, 0, 0
            else:
                tp, fp, fn = -1, 0, 0
        else:
            nonZeroGold += 1
            if gold.issubset(pred):
                covered += 1

        if len(pred) == 0:
            if len(gold) != 0:
                tp, fp, fn = -2, 0, 0

        num_TP.append(tp)
        num_FP.append(fp)
        num_FN.append(fn)

    return num_TP, num_FP, num_FN, nonZeroGold, covered


def _compute_metrics(
    num_TP: list, num_FP: list, num_FN: list, nonZeroGold: int, covered: int
):
    """
    Computes [micro, macro] x [Precision, Recall, F1], Private function
    """

    prec_micro, recall_micro, f1_micro = 0.0, 0.0, 0.0
    prec_macro, recall_macro, f1_macro = 0.0, 0.0, 0.0

    tp, fp, fn = [np.array(x).astype(float) for x in (num_TP, num_FP, num_FN)]

    prec_macro = np.nan_to_num(tp / (tp + fp))
    recall_macro = np.nan_to_num(tp / (tp + fn))
    idxs_special_case1 = np.where(tp == -1)[0]
    idxs_special_case2 = np.where(tp == -2)[0]
    prec_macro[idxs_special_case1] = 0.0  # type: ignore
    recall_macro[idxs_special_case1] = 1.0  # type: ignore
    prec_macro[idxs_special_case2] = 1.0  # type: ignore
    recall_macro[idxs_special_case2] = 0.0  # type: ignore
    f1_macro = np.nan_to_num(
        2 * prec_macro * recall_macro / (prec_macro + recall_macro)
    )
    tp[idxs_special_case1] = 0
    tp[idxs_special_case2] = 0

    prec_micro = np.nan_to_num(tp.sum() / (tp.sum() + fp.sum()))
    recall_micro = np.nan_to_num(tp.sum() / (tp.sum() + fn.sum()))
    f1_micro = np.nan_to_num(
        2 * prec_micro * recall_micro / (prec_micro + recall_micro)
    )

    all_metrics = {
        "micro": {"precision": prec_micro, "recall": recall_micro, "f1": f1_micro},
        "macro": {
            "precision": {
                "score": prec_macro.mean(),
            },
            "recall": {"score": recall_macro.mean()},
            "f1": {"score": f1_macro.mean()},
            # 'f1-QALD': f1_macro_qald
        },
        "Non-empty gold": nonZeroGold,
        "Covered": covered,
    }

    return all_metrics


def compute_p_r_f1_metrics(gold_answers: list, predicted_answers: list):
    """
    Use this to compute the metrics, expects list of sets
    gold_answers:        [ {q1_ans1, q1_ans2, ...}, {q2_ans1, ...}, ...]
    predicted_answers :  [ {q1_ans1, q1_ans2, ...}, {q2_ans1, ...}, ...]
    """
    tp, fp, fn, nonZeroGold, covered = _compute_confusion_mat(
        gold_answers, predicted_answers
    )
    return _compute_metrics(tp, fp, fn, nonZeroGold, covered)


def compute_score(gold_output, pred_output):
    # print('Internal Evaluation Metrics: ')
    system_answers_sets = []
    for answer in gold_output:
        system_answers_sets.append(set(answer))
    gold_answers_sets = []
    for answer in pred_output:
        gold_answers_sets.append(set(answer))

    metrics_results = compute_p_r_f1_metrics(system_answers_sets, gold_answers_sets)
    # print('Macro || Prec : {:.3f} | Recall : {:.3f} | F1 : {:.3f}'.format(
    #     metrics_results['macro']['precision']['score'], metrics_results['macro']['recall']['score'], metrics_results['macro']['f1']['score']))
    return (
        metrics_results["macro"]["precision"]["score"],
        metrics_results["macro"]["recall"]["score"],
        metrics_results["macro"]["f1"]["score"],
    )
