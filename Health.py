# Cell 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Cell 2: Load the Dataset
# Load the insurance dataset
data = pd.read_csv('insurance.csv')

# Cell 3: Data Preprocessing
# Separate features and target
X = data.drop('expenses', axis=1)
y = data['expenses']

# Split into train (80%) and test (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical columns
categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# Create a pipeline with preprocessor and linear regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Cell 4: Train the Model
# Fit the model on the training data
model.fit(X_train, y_train)

# Cell 5: Evaluate and Visualize Results
# Predict on test data
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: ${mae:.2f}")

# Check if the challenge is passed
if mae < 3500:
    print("Challenge passed: MAE is under $3500!")
else:
    print("Challenge failed: MAE is above $3500. Try improving the model.")

# Visualize actual vs predicted expenses
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Expenses ($)')
plt.ylabel('Predicted Expenses ($)')
plt.title('Actual vs Predicted Healthcare Expenses')
plt.tight_layout()
plt.show()