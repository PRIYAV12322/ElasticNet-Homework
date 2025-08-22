# ElasticNet Regression Example

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Generate a sample dataset
np.random.seed(42)
X = np.random.rand(100, 3)   # 100 samples, 3 features
y = 3*X[:,0] + 2*X[:,1] + X[:,2] + np.random.randn(100)  # linear relation with noise

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build ElasticNet model
# alpha = overall regularization strength
# l1_ratio = balance between Lasso (L1) and Ridge (L2)
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
