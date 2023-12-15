import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc




def plot_Histogram(lighter_scores, darker_scores,title,xlabel,ylabel):

    plt.figure(figsize=(10, 6))
    plt.hist(abs(lighter_scores), bins=20, alpha=0.5, color="turquoise", label="Outliers (FST I-IV Light)")
    plt.hist(abs(darker_scores), bins=20, alpha=0.5, color="darkorange", label="Outliers (FST V-VI Dark)")
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show the plot
    plt.show()
    
    

def plot_kde_graphs(df, labels_col, re_col, subset_labels, line_position=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for label in subset_labels:
        if (df[labels_col] == label).any():
            kde = gaussian_kde(df[re_col][df[labels_col] == label].values)
            df_subset = df[df[labels_col] == label]
            ax = df_subset[re_col].plot(kind="kde", ax=ax, label=f"Label {label}", alpha=0.5)

    if line_position is not None:
        ax.axvline(x=line_position, color='red', linestyle='--', label='Threshold')

    ax.legend()
    plt.show() 
    
    


def plot_roc_curve(in_dist_files, out_dist_files, labels, colors, title):
    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor('white')
    lw = 2

    for i in range(len(labels)):
        scores_in = np.loadtxt(in_dist_files[i])
        scores_out = np.loadtxt(out_dist_files[i])
        scores_in_train = scores_in[::2]
        scores_in_test = scores_in[1::2]
        scores_out_train = scores_out[::2]
        scores_out_test = scores_out[1::2]
        y_true = np.concatenate([np.repeat(1, scores_in_test.size), np.repeat(0, scores_out_test.size)])
        y_score = np.concatenate([scores_in_test, scores_out_test])
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        label = "{} (area = {:.2f})".format(labels[i], roc_auc)
        plt.plot(fpr, tpr, color=colors[i], lw=lw, label=label)

    plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate on Out-of-Distribution Set', fontsize=12)
    plt.ylabel('True Positive Rate on Validation Set', fontsize=12)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

    

  