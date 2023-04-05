import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises

kappa = 8
mu1 = 0
mu2 = 0.5

# Create an array of values between -pi and pi
x = np.linspace(-np.pi, np.pi, 200)

# Evaluate the von Mises distribution for each value of x
y1 = vonmises.pdf(x, kappa, loc=mu1)
y2 = vonmises.pdf(x, kappa, loc=mu2)

# Plot the distributions
plt.plot(x, y1, label=r"$\mu$ = {}".format(mu1))
plt.plot(x, y2, label=r"$\mu$ = {}".format(mu2))

# Add a legend and labels
plt.legend()
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Von Mises Distribution with Kappa = {}".format(kappa))

# Show the plot
plt.show()
