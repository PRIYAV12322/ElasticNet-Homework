import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Fit ElasticNet model
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

# Print results
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))

# ---------- Visualization ----------
plt.scatter(X, y, color="blue", label="Actual Data")   # scatter plot of real data
plt.plot(X, y_pred, color="red", linewidth=2, label="ElasticNet Prediction")  # regression line
plt.xlabel("X values")
plt.ylabel("y values")
plt.title("ElasticNet Regression Visualization")
plt.legend()
plt.show()
