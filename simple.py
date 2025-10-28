import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Generate a simple dataset
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0,0,0,0,0,1,1,1,1,1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

print(f"Predicted labels: {y_pred}, but expected: {y_test}")

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

x = np.array([[1.5], [50], [4.5]])
predictions = model.predict(x)
print(f"Predictions for new data {x}: {predictions}")