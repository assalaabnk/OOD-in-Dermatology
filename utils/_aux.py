import matplotlib.pyplot as plt

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


