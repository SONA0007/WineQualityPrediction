import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

# Load the wine dataset (Replace 'wine_dataset.csv' with your dataset file)
data = pd.read_csv('wine_dataset.csv')

# Filter out non-numeric columns (assuming 'quality' is the target column)
numerical_data = data.select_dtypes(include='number')

# Split the data into features (X) and target (y)
X = numerical_data.drop('quality', axis=1)
y = numerical_data['quality']

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'most_frequent'
X_imputed = imputer.fit_transform(X)

# Split the imputed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train a random forest regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print("Model R-squared score:", score)

# Save the trained model to a file
joblib.dump(model, 'wine_model.joblib')
