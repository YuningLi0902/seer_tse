import re
import time

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, confusion_matrix


def GetPredictCorrectness(result):
    pattern = r"(?i)it was [^a-zA-Z]*(correct|wrong)[^a-zA-Z]*"
    matches = re.findall(pattern, result)
    if matches:
        correctness = matches[-1]
        if correctness == "correct":
            return 1
        elif correctness == "wrong":
            return 0
    return -1


def evl(y_true, y_pred, threshold=0.5, return_raw=False):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)
    y_pred = [1 if p >= threshold else 0 for p in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    epsilon = 1e-6
    f1_denominator = prc + rc
    if f1_denominator < epsilon:
        f1 = float("nan")
    else:
        f1 = 2 * prc * rc / f1_denominator
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("TP:", tp)
    print("FP:", fp)
    print("TN:", tn)
    print("FN:", fn)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    if return_raw:
        return tn, fp, fn, tp, acc, prc, rc, f1, auc_, y_true, y_pred
    else:
        return tn, fp, fn, tp, acc, prc, rc, f1, auc_


def calculate_metrics(tp, tn, fp, fn):
    acc = (tp + tn) / (tp + tn + fp + fn)
    prc = tp / (tp + fp)
    rc = tp / (tp + fn)
    f1 = 2 * prc * rc / (prc + rc)
    return acc, prc, rc, f1


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
