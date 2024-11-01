# %% [markdown]
# # Regression
# 
# In  a regression problem, we  try to fit a line:
# 
# $$Y = v_0 + v_1 * X + \epsilon$$
# 
# We will try to adjust the intercept ($v_0$) and the slope ($v_1$) parameters of the line so that the sum of the distances of all points from the line becomes minimal.
# 
# We create a loss function that minimizes the difference between observed and predicted values, as a function of adjusting the line:
# 
# $$Loss = \frac{1}{n} \sum (Y - \hat{Y})^2$$
# $$Loss = \frac{1}{n} \sum (Y - v_0 + v_1 * X)^2$$
# 
# 
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


# %% [markdown]
# 1. Generate some data and plot them.

# %%
nb_samples = 200

# Generate data
X = np.linspace(-5, 5, nb_samples)  # Ensure exactly 200 points
Y = X + 2 + np.random.normal(0.0, 0.5, size=nb_samples)

# Plot data
plt.scatter(X, Y)
plt.show()


# %%
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
        g[0] += (v[0] + v[1] * X[i] - Y[i]) # partial derivative of loss with respect to v[0]
        g[1] += (v[0] + v[1] * X[i] - Y[i]) * X[i] # partial derivative of loss with respect to v[1]
    return g




# %%
result=minimize(fun=loss, x0=[0.0, 0.0], jac=gradient, method='Powell')
print(result)

# %% [markdown]
# ### Let's do the same with scikit learn
# 
# We need to first reshape X

# %%
X.shape
Y.shape

X= X.reshape(-1,1)
X.shape

# %%
reg = LinearRegression().fit(X, Y)
reg.score(X, Y)

# %%
print(reg.coef_)
print(reg.intercept_)

# %% [markdown]
# 
# # Ridge regression (L1 regression) , Lasso regression (L2 regression), and Elastic Net
# 
# Simple regression can be prone to over fitting. This is why we add a regularisation parameter alpha that "reverses" the tightening of the fit. Adding $\alpha$ will also allow us to identify which predictors are relevant for multiple linear regression.
# 
# Again, we are trying to fit the line:
# 
# $$Y = v_0 + v_1 * X_1 + v_1 * X_2 ... + \epsilon$$
# 
# $$Y = v_0 + v * X + \epsilon$$
# 
# #### L1 Regression:
# 
# $$Loss = \frac{1}{n} \sum (Y - \widehat{Y}) + \lambda \sum_1^n |v_i|$$
# 
# $$Loss = \frac{1}{n} \sum (Y - v_0 + v * X) + \lambda \sum_1^n |v_i|$$
# 
# #### L2 Regression:
# 
# $$Loss = \frac{1}{n} \sum (Y - \widehat{Y}) +  \lambda \sum_1^n v_i^{2}$$
# 
# $$Loss = \frac{1}{n} \sum (Y - v_0 + v * X) +  \lambda \sum_1^n v_i^{2}$$
# 
# ### Elastic net
# 
# Elastic nets are a trade off between ridge and lasso regression. An additional toggle parameter $\rho$ is being used to toggle between ridge and lasso regression.
# 
# $$Loss = \frac{1}{n} \sum (Y - v_0 + v * X) +  \alpha * ( \rho \sum_1^n |v_i| + (1-\rho) \sum_1^n v_i^{2})$$

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Number of samples and features
nb_samples = 200
nb_features = 10

# Generate synthetic data with correlated features
np.random.seed(42)  # For reproducibility
X_base = np.random.randn(nb_samples, 1)
X = np.hstack([X_base + 0.1 * np.random.randn(nb_samples, 1) for _ in range(nb_features)])

# True coefficients, only a few are non-zero (sparse)
true_coefficients = np.zeros(nb_features)
true_coefficients[:3] = [1.5, -2.0, 3.0]  # Only the first three features are relevant

# Generate target variable with some noise
Y = X @ true_coefficients + np.random.normal(0.0, 0.5, size=nb_samples)

# Plotting the correlation matrix to show multicollinearity
plt.imshow(np.corrcoef(X, rowvar=False), cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title("Feature Correlation Matrix")
plt.show()

# L1 (Lasso) Loss function
def loss_l1(v, alpha=1.0):
    """Calculate the L1 loss for linear regression with regularization."""
    e = 0.0
    for i in range(nb_samples):
        e += np.square(np.dot(X[i], v) - Y[i])
    # L1 regularization term
    l1_penalty = alpha * np.sum(np.abs(v))
    return 0.5 * e + l1_penalty

# L2 (Ridge) Loss function
def loss_l2(v, alpha=1.0):
    """Calculate the L2 loss for linear regression with regularization."""
    e = 0.0
    for i in range(nb_samples):
        e += np.square(np.dot(X[i], v) - Y[i])
    # L2 regularization term
    l2_penalty = alpha * np.sum(v**2)
    return 0.5 * e + l2_penalty

# Elastic Net Loss function
def loss_elastic_net(v, alpha=1.0, l1_ratio=0.5):
    """Calculate the elastic net loss for linear regression with regularization."""
    e = 0.0
    for i in range(nb_samples):
        e += np.square(np.dot(X[i], v) - Y[i])
    # Elastic net regularization term
    l1_penalty = l1_ratio * np.sum(np.abs(v))
    l2_penalty = (1 - l1_ratio) * np.sum(v**2)
    elastic_penalty = alpha * (l1_penalty + l2_penalty)
    return 0.5 * e + elastic_penalty

# Perform optimization for L1
result_l1 = minimize(fun=loss_l1, x0=np.zeros(nb_features), method='Powell')
print("L1 Regression Result Coefficients:", result_l1.x)

# Perform optimization for L2
result_l2 = minimize(fun=loss_l2, x0=np.zeros(nb_features), method='Powell')
print("L2 Regression Result Coefficients:", result_l2.x)

# Perform optimization for Elastic Net
result_elastic_net = minimize(fun=loss_elastic_net, x0=np.zeros(nb_features), method='Powell')
print("Elastic Net Regression Result Coefficients:", result_elastic_net.x)

# Plotting the results
plt.plot(true_coefficients, label='True Coefficients', linestyle='--', marker='o')
plt.plot(result_l1.x, label='L1 (Lasso) Coefficients', linestyle='--', marker='x')
plt.plot(result_l2.x, label='L2 (Ridge) Coefficients', linestyle='--', marker='s')
plt.plot(result_elastic_net.x, label='Elastic Net Coefficients', linestyle='--', marker='d')
plt.title("Comparison of Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.show()


# %% [markdown]
# ### Let's do the same with sklearn

# %%
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Lasso (L1 regularization)
lasso = Lasso(alpha=1.0, max_iter=10000)
lasso.fit(X, Y)
print("Lasso (L1) Regression Coefficients:", lasso.coef_)

# Ridge (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X, Y)
print("Ridge (L2) Regression Coefficients:", ridge.coef_)

# Elastic Net (combined L1 and L2 regularization)
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
elastic_net.fit(X, Y)
print("Elastic Net Regression Coefficients:", elastic_net.coef_)

# Plotting the results
plt.plot(true_coefficients, label='True Coefficients', linestyle='--', marker='o')
plt.plot(lasso.coef_, label='Lasso (L1) Coefficients', linestyle='--', marker='x')
plt.plot(ridge.coef_, label='Ridge (L2) Coefficients', linestyle='--', marker='s')
plt.plot(elastic_net.coef_, label='Elastic Net Coefficients', linestyle='--', marker='d')
plt.title("Comparison of Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.show()


