import numpy as np
import matplotlib.pyplot as plt

nb_samples = 200

X = np.arange(-5, 5, 0.05)

Y = X + 2

Y += np.random.normal(0.0, 0.5, size = nb_samples)

plt.scatter(X, Y)
plt.show()

def loss(v):
    e = 0.0
    for i in range(nb_samples):
        e += np.square(v[0] + v[1]*X[i] - Y[i])
    return 0.5*e
    