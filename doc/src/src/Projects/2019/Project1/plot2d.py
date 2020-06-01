from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm # This is needed to define a colormap
import numpy as np
from random import random, seed

# Define own function to generate some height values
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Make data.
n_row = 100
n_col = 1000

ax_row = np.random.randn(n_row)
ax_col = np.random.randn(n_col)

# If you wish to plot the data, remember to sort the axes such that the surface
# is rendered correctly

sort_inds_row = np.argsort(ax_row) # This functions returns the indices to
                                   # ax_row that makes ax_row sorted
                                   # (see the declaration of ax_row_sorted on how to use the indices)
sort_inds_col = np.argsort(ax_col)

ax_row_sorted = ax_row[sort_inds_row]
ax_col_sorted = ax_col[sort_inds_col]

# Grid the data.
# This must be done such that surf (surface plot) understands which (x,y)- point corresponds to which z-value
# Meshgrid returns matrices where the first matrix is ax_column repeated along the rows (vertical direction),
# where the second matrix has ax_row repeated along the columns (horizontal direction) .
# Each pair (rowmat[i,j], colmat[i,j]) for all i,j constitues a coordinate in
# the plane in the rectangular domain [min(ax_row), max(ax_row)]x[min(ax_column), max(ax_column)]
colmat, rowmat = np.meshgrid(ax_col_sorted, ax_row_sorted)

# This evaluates the height associated for each pair of coordinate made from np.meshgrid
z = FrankeFunction(rowmat, colmat)

"""
If you wish to make some predictons on z using some regression method and plot the result,
it is possible to do so by predicting every pair of (rowmat[i,j], colmat[i,j]).

One could think of the pair  (rowmat[i,j], colmat[i,j]) as x-and y-coordinates in the plane.
When you have your parameters, apply them to every  (rowmat[i,j], colmat[i,j]) - pair.
This could for instance be done by a double for loop, where you perform the predictions
within the innermost loop.

Every prediction can then be stored in a matrix, e.g z_pred
where z_pred[i,j] corresponds to the prediction of (rowmat[i,j], colmat[i,j]),
that can be visualized in the same manner as z in the code below.
"""

# Plot the generated surface.
fig = plt.figure() # Starts a new window

# specify that we are plotting in 3D
ax = fig.gca(projection='3d')

surf = ax.plot_surface(rowmat, colmat, z, linewidth = 0, antialiased = False, cmap=cm.viridis)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Add a color bar that shows which color each value is mapped to.
fig.colorbar(surf)

# Show the plot
plt.show()
