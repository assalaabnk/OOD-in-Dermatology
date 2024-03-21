import numpy as np 
import pandas as pd 
from sklearn.metrics import roc_curve, auc, f1_score



def calculate_metrics(scores_in, scores_out, optimal_delta, num_subsets=4):
    auc_values = []
    f1_values = []

    for i in range(num_subsets):
        np.random.shuffle(scores_in)
        np.random.shuffle(scores_out)

        scores_in_train = scores_in[:len(scores_in) // 2]
        scores_in_test = scores_in[len(scores_in) // 2:]
        scores_out_train = scores_out[:len(scores_out) // 2]
        scores_out_test = scores_out[len(scores_out) // 2:]

        y_true = np.concatenate([np.repeat(1, scores_in_test.size), np.repeat(0, scores_out_test.size)])
        y_score = np.concatenate([scores_in_test, scores_out_test])

        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)

        y_true2 = np.concatenate([np.repeat(0, len(scores_in)), np.repeat(1, len(scores_out))])
        y_score2 = np.concatenate([scores_in, scores_out])
        y_pred = (y_score2 < optimal_delta).astype(int)
        f1 = f1_score(y_true2, y_pred)
        f1_values.append(f1)

    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)
    mean_f1 = np.mean(f1_values)
    std_f1 = np.std(f1_values)

    print("Mean AUC: {:.4f}, Mean F1 Score: {:.4f}".format(mean_auc, mean_f1))
    print("Std Deviation of AUC: {:.4f}, Std Deviation of F1 Score: {:.4f}".format(std_auc, std_f1))