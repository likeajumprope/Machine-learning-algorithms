import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

nb_samples = 200

X = np.arange(-5, 5, 0.05)

Y = X + 2

Y += np.random.normal(0.0, 0.5, size = nb_samples)

plt.scatter(X, Y)
#plt.show()


# functions
def loss(v):
    """_summary_

    Args:
        v (np_array): numpy array containing fitted parameters 

    Returns:
        scalar: summed loss
    """
    e = 0.0
    for i in range(nb_samples):
        e += np.square(v[0] + v[1] * X[i] - Y[i])
    return 0.5 * e
    
    
def gradient(v):
    g = np.zeros(shape=2)
    for i in range(nb_samples):
        g[0] += (v[0] + v[1] * X[i] - Y[i])
        g[1] += (v[0] + v[1] * X[i] - Y[i] * X[i])
    return g


result=minimize(fun=loss, x0=[0.0, 0.0], jac=gradient, method='Powell')
print(result)

# sklearn

X.shape
Y.shape

X= X.reshape(-1,1)
X.shape

reg = LinearRegression().fit(X, Y)
reg.score(X, Y)

print(reg.coef_)
print(reg.intercept_)