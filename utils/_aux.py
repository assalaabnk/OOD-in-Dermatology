import numpy as np
import visdom
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable




# ===== Visualization Helpers =====
def create_plot_window(vis, xlabel, ylabel, title):
    return vis.line(
        X=np.array([1]),
        Y=np.array([np.nan]),
        opts=dict(xlabel=xlabel, ylabel=ylabel, title=title),
    )


def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_re(groups, threshold_fixed):
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(
            group.index,
            group.re,
            marker="o",
            ms=3.5,
            linestyle="",
            label="ISIC 2019" if name == 0 else "FST",
        )
    ax.hlines(
        threshold_fixed,
        ax.get_xlim()[0],
        ax.get_xlim()[1],
        colors="r",
        zorder=100,
        label="Threshold",
    )
    ax.legend()
    plt.title(
        "Reconstruction error for in-distribution and out-of-distribution samples"
    )
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.show()


# ===== PyTorch Helpers =====
def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

## Auxiliary functions for the ipynb

def show_samples(images, groundtruth):
    """
    Show examples of ISIC 2019 along with
    condition classification.
    Input: paths, labels
    Output: grid images
    """

    f, axarr = plt.subplots(3, 4)
    f.suptitle("Samples from ISIC 2019")
    curr_row = 0
    for index, name in enumerate(images[:12]):
        # print(name.stem)
        a = plt.imread(name)
        # find the column by taking the current index modulo 3
        col = index % 3
        # plot on relevant subplot
        axarr[col, curr_row].imshow(a)
        axarr[col, curr_row].text(
            5,
            5,
            str(groundtruth.loc[name.stem].idxmax(axis=0)),
            bbox={"facecolor": "white"},
        )
        if col == 2:
            curr_row += 1

    f.tight_layout()
    return f

## Auxiliary functions for the ipynb 


def plot_pca(X_pca, y):
    """
    Plot a 2D PCA for OOD samples
    """
    plt.figure()
    colors = ["navy", "darkorange", "turquoise"]  # Adjust colors for the three classes
    lw = 2

    for color, i, target_name in zip(colors, [0, 2, 3], ["isic 2019", "FST V-VI", "FST I-IV"]):
        plt.scatter(
            X_pca[y == i, 0],
            X_pca[y == i, 1],
            color=color,
            alpha=0.5,
            lw=lw,
            label=target_name,
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA ")
    plt.xlabel("pc1")  # Set x-axis label
    plt.ylabel("pc2")  # Set y-axis label