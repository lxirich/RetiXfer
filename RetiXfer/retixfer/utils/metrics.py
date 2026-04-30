import os
import numpy as np
import json

from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, f1_score, recall_score, average_precision_score


def evaluate(refs, preds, task="classification", dataset=''):
    if task == "classification":
        metrics = classification_metrics(refs, preds, dataset)
        print('Metrics: aca=%2.5f - kappa=%2.3f - macro f1=%2.3f - auc=%2.3f  - aupr=%2.3f'
                % (metrics["aca"], metrics["kappa"],metrics["f1_avg"],metrics["auc_avg"],metrics["aupr_avg"]))
    else:
        metrics = {}
    return metrics


def au_prc(true_mask, pred_mask):
    precision, recall, threshold = precision_recall_curve(true_mask, pred_mask)
    au_prc = auc(recall, precision)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    f1[np.isnan(f1)] = 0
    th = threshold[np.argmax(f1)]

    return au_prc, th


def specificity(refs, preds):
    cm = confusion_matrix(refs, preds)
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    return specificity


def classification_metrics(refs, preds, dataset):
    aupr_class = [np.round(average_precision_score(refs == i, preds[:, i]), 3) for i in np.unique(refs)]
    
    k = np.round(cohen_kappa_score(refs, np.argmax(preds, -1), weights="quadratic"), 3)                             

    cm = confusion_matrix(refs, np.argmax(preds, -1))                                                                      
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))

    acc_class = list(np.round(np.diag(cm_norm), 3))
    aca = np.round(np.mean(np.diag(cm_norm)), 3)

    recall_class = [np.round(recall_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]      

    specificity_class = [np.round(specificity(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]   

    auc_class = [np.round(roc_auc_score(refs == i, preds[:, i]), 3) for i in np.unique(refs)]                       

    f1_class = [np.round(f1_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]               

    class_acc = accuracy_score(refs, np.argmax(preds, -1))

    metrics = {"aupr": aupr_class, "aupr_avg": np.mean(aupr_class), "aca": aca ,"acc_class": acc_class,
                "kappa": k, "auc_class": auc_class, "auc_avg": np.mean(auc_class),
                "f1_class": f1_class, "f1_avg": np.mean(f1_class),
                "sensitivity_class": recall_class, "sensitivity_avg": np.mean(recall_class),
                "specificity_class": specificity_class, "specificity_avg": np.mean(specificity_class),
                "cm": cm, "cm_norm": cm_norm}
    return metrics


def average_folds_results(list_folds_results, task):
    metrics_name = list(list_folds_results[0].keys())               

    out = {}
    for iMetric in metrics_name:
        values = np.concatenate([np.expand_dims(np.array(iFold[iMetric]), -1) for iFold in list_folds_results], -1)
        out[(iMetric + "_avg")] = np.round(np.mean(values, -1), 3).tolist()
        out[(iMetric + "_std")] = np.round(np.std(values, -1), 3).tolist()

    if task == "classification":
        print('Metrics: aca=%2.3f(%2.3f)  - kappa=%2.3f(%2.3f) - macro f1=%2.3f(%2.3f) - auc=%2.3f(%2.3f) - aupr=%2.3f(%2.3f)' 
               % (
            out["aca_avg"], out["aca_std"],
            out["kappa_avg"], out["kappa_std"],
            out["f1_avg_avg"], out["f1_avg_std"],out["auc_avg_avg"], out["auc_avg_std"],out["aupr_avg_avg"], out["aupr_avg_std"]
            ))

    return out


def save_results(metrics, out_path, id_experiment=None, id_metrics=None, save_model=False, weights=None):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    if id_experiment is None:
        id_experiment = "experiment" + str(np.random.rand())
    else:
        id_experiment = id_experiment
    if not os.path.isdir(out_path + id_experiment):
        os.mkdir(out_path + id_experiment)

    with open(out_path + id_experiment + '/metrics_' + id_metrics + '.json', 'w') as fp:
        json.dump(metrics, fp)

    if save_model:
        import torch
        for i in range(len(weights)):
            torch.save(weights[i], out_path + id_experiment + '/weights_' + str(i) + '.pth')