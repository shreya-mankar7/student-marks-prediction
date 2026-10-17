import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create the dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [20, 25, 35, 45, 50, 60, 70, 85]
}

df = pd.DataFrame(data)

print("Student Marks Dataset:")
print(df)

# Separate input and target
X = df[["Hours"]]
y = df["Marks"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test marks
y_pred = model.predict(X_test)

print("\nActual Marks:", list(y_test))
print("Predicted Marks:", y_pred.round(2))

# Evaluate the model
print("\nMAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

# Predict marks for 9 hours of study
new_student = pd.DataFrame({"Hours": [9]})
prediction = model.predict(new_student)

print("\nPredicted marks for 9 hours:", round(prediction[0], 2))

# Plot the results
plt.scatter(X["Hours"], y, label="Actual Marks")
plt.plot(X["Hours"], model.predict(X), label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Student Marks Prediction")
plt.legend()
plt.show()
