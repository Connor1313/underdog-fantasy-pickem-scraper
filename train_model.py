import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load the training data
X_train = pd.read_csv("train_features.csv")
Y_train = pd.read_csv("train_targets.csv")

# Define the model
rf = RandomForestRegressor(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Save the trained model
joblib.dump(best_model, "optimized_model.pkl")

# Print results
print(f"Best Model Parameters: {grid_search.best_params_}")
print("Model trained and saved to optimized_model.pkl")
