import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class ModelTester:
    def __init__(self, model_path, test_data_path, actuals_path=None):
        """
        Initialize the ModelTester.
        :param model_path: Path to the trained model (e.g., "optimized_model.pkl")
        :param test_data_path: Path to test data (e.g., "test_features.csv")
        :param actuals_path: Path to actual values for evaluation (optional)
        """
        self.model = joblib.load(model_path)
        self.test_data = pd.read_csv(test_data_path)
        self.actuals = pd.read_csv(actuals_path) if actuals_path else None

    def predict(self):
        """Runs predictions on the test dataset."""
        predictions = self.model.predict(self.test_data)
        return predictions

    def evaluate(self, predictions):
        """Evaluates predictions against actual values if available."""
        if self.actuals is None:
            print("No actual values provided. Only predictions will be returned.")
            return None
        
        # Convert to DataFrame for comparison
        predictions_df = pd.DataFrame(predictions, columns=["pred_kills", "pred_assists", "pred_deaths"])
        results_df = pd.concat([self.actuals.reset_index(drop=True), predictions_df], axis=1)

        # Calculate error metrics
        mae = mean_absolute_error(self.actuals, predictions)
        mse = mean_squared_error(self.actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.actuals, predictions)

        # Print evaluation metrics
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R² Score: {r2}")

        return results_df

# Example Usage
if __name__ == "__main__":
    # Update these paths based on the trained model you want to test
    MODEL_PATH = "optimized_model.pkl"
    TEST_DATA_PATH = "test_features.csv"
    ACTUALS_PATH = "test_targets.csv"

    tester = ModelTester(MODEL_PATH, TEST_DATA_PATH, ACTUALS_PATH)
    
    # Run predictions
    preds = tester.predict()
    
    # Evaluate results (if actuals are available)
    results = tester.evaluate(preds)
    
    # Save the results to a CSV file
    if results is not None:
        results.to_csv("test_results.csv", index=False)
        print("Test results saved to test_results.csv.")
