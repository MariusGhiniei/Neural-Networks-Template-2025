import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from timed_decorator.simple_timed import timed
from typing import Tuple

predicted = np.array([
    1,1,1,0,1,0,1,1,0,0
])
actual = np.array([
    1,1,1,1,0,0,1,0,0,0
])

big_size = 5000000
big_actual = np.repeat(actual, big_size)
big_predicted = np.repeat(predicted, big_size)

@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_sklearn(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    return tp, fp, fn, tn


@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_numpy(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    gt_bool = gt > 0
    pred_bool = pred > 0

    tp = np.sum((pred_bool == True) & (gt_bool == True))
    fp = np.sum((pred_bool == True) & (gt_bool == False))
    fn = np.sum((pred_bool == False) & (gt_bool == True))
    tn = np.sum((pred_bool == False) & (gt_bool == False))
    return tp, fp, fn, tn
    #raise NotImplementedError()


assert tp_fp_fn_tn_sklearn(actual, predicted) == tp_fp_fn_tn_numpy(actual, predicted)

rez_1 = tp_fp_fn_tn_sklearn(big_actual, big_predicted)
rez_2 = tp_fp_fn_tn_numpy(big_actual, big_predicted)

assert rez_1 == rez_2

# Accuracy

@timed(use_seconds=True, show_args=True)
def accuracy_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return accuracy_score(gt, pred)


@timed(use_seconds=True, show_args=True)
def accuracy_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_bool = gt > 0
    pred_bool = pred > 0

    tp = np.sum((pred_bool == True) & (gt_bool == True))
    tn = np.sum((pred_bool == False) & (gt_bool == False))

    return (tp + tn) / pred_bool.size


assert accuracy_sklearn(actual, predicted) == accuracy_numpy(actual, predicted)

rez_1 = accuracy_sklearn(big_actual, big_predicted)
rez_2 = accuracy_numpy(big_actual, big_predicted)

print("Accuracy from sklearn: ", rez_1)
print("Accuracy from numpy: ", rez_2)

assert np.isclose(rez_1, rez_2)

#F1 - Score

#Precision = tp/(tp+fp)
#Recall = tp/(tp+fn)
#F1-score = 2 * Precision * Recall / (Precision + recall)

@timed(use_seconds=True, show_args=True)
def f1_score_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return f1_score(gt, pred)


@timed(use_seconds=True, show_args=True)
def f1_score_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_bool = gt > 0
    pred_bool = pred > 0

    tp = np.sum((pred_bool == True) & (gt_bool == True))
    fp = np.sum((pred_bool == True) & (gt_bool == False))
    fn = np.sum((pred_bool == False) & (gt_bool == True))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return   2 * precision * recall / (precision + recall + 0.0000001)

assert np.isclose(f1_score_sklearn(actual, predicted), f1_score_numpy(actual, predicted))

rez_1 = f1_score_sklearn(big_actual, big_predicted)
rez_2 = f1_score_numpy(big_actual, big_predicted)

assert np.isclose(rez_1, rez_2)