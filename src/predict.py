import joblib
import numpy as np

# Load model
model = joblib.load("model.joblib")

# Dummy input (10 features like diabetes dataset)
sample = np.random.rand(1, 10)

# Predict
prediction = model.predict(sample)
print("Prediction:", prediction)
