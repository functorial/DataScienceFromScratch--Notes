import numpy as np
import random
from matplotlib import pyplot as plt 
from stats import inverse_normal_cdf

# inputs a list of n data vectors 
# returns a n x n matrix
# i,j-th entry is Correlation(d_i, d_j)
correlation_matrix = np.corrcoef

# A more visual approach using subplots:
# `corr_data` is a list of vectors
xs = [[inverse_normal_cdf(random.random()) for _ in range(200)] for _ in range(4)]
corr_data = [[p  for p in x] for (i,x) in zip(range(4), xs)]
num_vectors = len(corr_data)

fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
    for j in range(num_vectors):

        # scatter column_j on the x-axis vs column_i on the y axis
        if i != j: ax[i][j].scatter(corr_data[j], corr_data[i])

        # unless i == j, in which case show the series name
        else: 
            ax[i][j].annotate("series " + str(i), (0.5, 0.5), xycoords='axes fraction', ha="center", va= "center")

        # Then hide axis labels except left and bottom charts
        if i < num_vectors - 1:
            ax[i][j].xaxis.set_visible(False)
        if j > 0:
            ax[i][j].yaxis.set_visible(False)
# Fix the bottom-right and top-left axis labels, which are wrong because
# their charts only have text in them
ax[-1][-1].set_xlim(ax[0][1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())

plt.show()