import pandas as pd 
import numpy as np
%matplotlib inline
import os
import matplotlib.pyplot as plt





# Define marker styles for each model
marker_styles = {
    'One_SVM': 'o',
    'Isolation_Forest': 'X',
    'Auto_Encoder': '^',
    #'ODIN': 'D',
    #'NN_Softmax': 's'
}

# Define marker sizes for each model (optional customization)
marker_sizes = {
    'One_SVM': 100,
    'Isolation_Forest': 100,
    'Auto_Encoder': 100,
   # 'ODIN': 100,
    #'NN_Softmax': 100
}


def Scatter_Plot(x, y,  labels,x_label='x',y_label='y',X='x',Y='y',P='p'):
    # linear regression parameters
    slope, intercept = np.polyfit(x, y, 1)
    # correlation coefficient
    Corr = np.corrcoef(x, y)[0, 1]

    # Create scatter plot
    #plt.figure(figsize=(8, 6))
    #plt.scatter(x, y, marker='o', color='b')
    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # Scatter plot with unique markers and sizes
    for label in np.unique(labels):
        plt.scatter(
            x[labels == label],
            y[labels == label],
            marker=marker_styles[label],
            s=marker_sizes[label],  # Adjust marker size
            label=label
        )

    # Draw the regression line
    plt.plot(x, slope * x + intercept, color='r', label='Regression Line')

    # Labels
    plt.xlabel(x_label + ' (' + X + ')', fontsize=14)
    plt.ylabel(y_label + ' (' + Y + ')', fontsize=14)
    plt.title('Correlation between ' + X + ' and ' + Y+ ' Fitz17k '+P+'%', fontsize=16)

    plt.legend(fontsize=12)

    # Add correlation coefficient to plot
    plt.text(0.85, 0.7, f'Corr: {Corr:.2f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1, edgecolor='none'))


    plt.grid(True)
    plt.show()