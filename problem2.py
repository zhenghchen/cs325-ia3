import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# Load and prepare the data
data = pd.read_csv("Corvallis.csv", delimiter=";")

# We use the 'day' column (days since May 1, 1952) as d_i and the 'average' column as T_i.
d_values = data['day.1'].values
T_values = data['average'].values

# Define the periods for the two sinusoidal components.
P_season = 365.25              # period for the annual cycle
P_solar = 365.25 * 10.7        # period for the solar cycle

### Code for linear programming ###


# x0, x1, x2, x3, x4, x5, E
# minimize E
c = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (7, 1), 'd')

G_rows = []
h_rows = []

for d, T in zip(d_values, T_values):
    seasonal_cos = np.cos(2 * np.pi * d / P_season)
    seasonal_sin = np.sin(2 * np.pi * d / P_season)
    solar_cos = np.cos(2 * np.pi * d / P_solar)
    solar_sin = np.sin(2 * np.pi * d / P_solar)

    # Constraint 1:
    # x0 + x1*d + x2*seasonal_cos + x3*seasonal_sin + x4*solar_cos + x5*solar_sin - T <= E
    #
    # x0 + x1*d + x2*seasonal_cos + x3*seasonal_sin + x4*solar_cos + x5*solar_sin - E <= T
    G_rows.append([
        1.0,
        d,
        seasonal_cos,
        seasonal_sin,
        solar_cos,
        solar_sin,
        -1.0
    ])
    h_rows.append(T)

    # Constraint 2:
    # -(x0 + x1*d + x2*seasonal_cos + x3*seasonal_sin + x4*solar_cos + x5*solar_sin - T) <= E
    #
    # -x0 - x1*d - x2*seasonal_cos - x3*seasonal_sin - x4*solar_cos - x5*solar_sin - E <= -T
    G_rows.append([
        -1.0,
        -d,
        -seasonal_cos,
        -seasonal_sin,
        -solar_cos,
        -solar_sin,
        -1.0
    ])
    h_rows.append(-T)

# E >= 0
# -E <= 0
G_rows.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
h_rows.append(0.0)

G = matrix(np.array(G_rows, dtype='d'))
h = matrix(np.array(h_rows, dtype='d'))

sol = solvers.lp(c, G, h)

# solution
x0 = sol['x'][0]
x1 = sol['x'][1]
x2 = sol['x'][2]
x3 = sol['x'][3]
x4 = sol['x'][4]
x5 = sol['x'][5]
E = sol['x'][6]


###################################

print("Optimal solution found:")
print(f"x0 = {x0}")
print(f"x1 = {x1}  (daily drift in °C)")
print(f"x2 = {x2}")
print(f"x3 = {x3}")
print(f"x4 = {x4}")
print(f"x5 = {x5}")
print(f"Minimum maximum absolute deviation (E) = {E}")
print(f"Estimated annual drift (x1 * 365.25) = {x1 * 365.25} °C/year")

# Plot the D and T values
plt.figure(figsize=(14, 6))
plt.plot(d_values, T_values, 'ro', label="Data points")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")

# Plot the fitted model
d_values_sorted = np.sort(d_values)
T_model_values = [x0 + x1 * d + x2 * np.cos(2 * np.pi * d / P_season) + x3 * np.sin(2 * np.pi * d / P_season) + x4 * np.cos(2 * np.pi * d / P_solar) + x5 * np.sin(2 * np.pi * d / P_solar) for d in d_values_sorted]
plt.plot(d_values_sorted, T_model_values, 'b-', label="Fitted model")

# Plot the linear trend
d_values_sorted = np.sort(d_values)
T_trend_values = [x0 + x1 * d for d in d_values_sorted]
plt.plot(d_values_sorted, T_trend_values, 'g-', label="Linear trend")

plt.title("Temperature Fit with Seasonal and Solar Components")
plt.legend()
plt.show()

# Save the plot
plt.savefig("temperature_fit.png")
print("Plot saved as 'temperature_fit.png'")
