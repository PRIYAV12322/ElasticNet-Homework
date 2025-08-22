import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------------------
# Generate synthetic dataset
# ------------------------------
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(100)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# ElasticNet Regression
# ------------------------------
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------------
# Print Results
# ------------------------------
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ------------------------------
# Visualization
# ------------------------------
plt.scatter(X_test, y_test, color="blue", label="Actual data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("ElasticNet Regression")
plt.legend()

# Save and Show
plt.savefig("elasticnet_plot.png")  # Save the plot in repo
plt.show()
