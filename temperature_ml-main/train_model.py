import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

X = np.array([
    [0], [10], [20],
    [30], [40], [50]
])

y = np.array([
    32, 50, 68,
    86, 104, 122
])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model saved!")