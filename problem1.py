
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import numpy as np

# Define the data points
points = [(1, 3), (2, 5), (3, 7), (5, 11), (7, 14), (8, 15), (10, 19)]

### Code for linear programming ###
n = len(points)

c = [0, 0, 1.0]

G_rows = []
h_rows = []

for (xi, yi) in points:
    # Constraint 1: axi + b  -yi <= z
    # axi + b - z <= yi
    G_rows.append([xi, 1, -1])
    h_rows.append(yi)

    # Constraint 2: -(axi + b - yi) <= z
    # -(axi) -b -z <= -yi
    G_rows.append([-xi, -1, -1])
    h_rows.append(-yi)

G = matrix(np.array(G_rows, dtype='d'))
h = matrix(np.array(h_rows, dtype='d'))
c = matrix(c, (3, 1), 'd')
sol = solvers.lp(c, G, h)

a = sol['x'][0]
b = sol['x'][1]
E = sol['x'][2]


###################################

# display the results
print("Optimal a:", a)
print("Optimal b:", b)
print("Optimal t (max absolute deviation):", E)

# Plot the data points and the regression line
plt.figure()
plt.plot([x for (x, y) in points], [y for (x, y) in points], 'ro', label="Data points")
plt.plot([x for (x, y) in points], [a* x + b for (x, y) in points], 'b-', label="Regression line")
plt.legend()
plt.show()

# Save the plot
plt.savefig("regression_plot.png")
print("Plot saved as regression_plot.png")

