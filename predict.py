import pandas as pd
import joblib
import numpy as np

class UnderdogPredictor:
    def __init__(self, model_path, underdog_data_path, training_data_path):
        """
        Initialize the predictor.
        :param model_path: Path to the trained model (e.g., "optimized_model.pkl")
        :param underdog_data_path: Path to Underdog Fantasy lines (e.g., "underdog_esports_data.csv")
        :param training_data_path: Path to full Oracle's Elixir dataset (e.g., "oracle_elixir.csv")
        """
        self.model = joblib.load(model_path)
        self.underdog_data = pd.read_csv(underdog_data_path)  # Underdog betting lines
        self.training_data = pd.read_csv(training_data_path)  # Oracle's Elixir full dataset

        # Extract feature names from training data (excluding targets)
        self.features = [col for col in self.training_data.columns if col not in ["kills", "assists", "deaths"]]

    def preprocess_underdog_data(self):
        """Prepare Underdog Fantasy data for model predictions."""
        # Keep only League of Legends data
        lol_data = self.underdog_data[self.underdog_data["sport_id"] == "LOL"]

        # Remove unwanted columns like fantasy points
        lol_data = lol_data.drop(columns=["fantasy_points"], errors="ignore")

        # Standardize stat names (remove map-specific details)
        lol_data["stat_name"] = lol_data["stat_name"].str.replace(r"_on_map_\d+", "", regex=True)

        # Aggregate by full_name & stat_name to sum all maps
        lol_data = lol_data.groupby(["full_name", "stat_name"], as_index=False)["stat_value"].sum()

        # Pivot table to structure it properly
        lol_data = lol_data.pivot(index="full_name", columns="stat_name", values="stat_value").reset_index()

        return lol_data

    def match_training_features(self, X_underdog):
        """Ensure test features match training features in order and format."""
        # Get missing features
        missing_features = [col for col in self.features if col not in X_underdog.columns]

        # Fill missing features with the median value from training data
        for col in missing_features:
            X_underdog[col] = np.nan  # Fill missing with NaN

        # Fill NaNs with column median values
        X_underdog = X_underdog.fillna(self.training_data.median())

        # Ensure feature order matches training
        X_underdog = X_underdog[self.features]

        return X_underdog

    def predict(self):
        """Predict player performance (kills, assists, deaths) based on Underdog lines."""
        lol_underdog = self.preprocess_underdog_data()

        # Drop player names before prediction
        X_underdog = lol_underdog.drop(columns=["full_name"], errors="ignore")

        # Match training features
        X_underdog = self.match_training_features(X_underdog)

        # Predict
        predictions = self.model.predict(X_underdog)

        # Convert predictions into a DataFrame
        predictions_df = pd.DataFrame(
            predictions, columns=["pred_kills", "pred_assists", "pred_deaths"]
        )

        # Merge player names back
        results_df = pd.concat([lol_underdog[["full_name"]], predictions_df], axis=1)

        return results_df

    def compare_to_underdog_lines(self, predictions_df):
        """Compare model predictions to actual Underdog Fantasy betting lines."""
        lol_underdog = self.preprocess_underdog_data()

        # Merge predictions with actual Underdog lines
        results_df = lol_underdog.merge(predictions_df, on="full_name", how="left")

        # Print results
        for index, row in results_df.iterrows():
            print(f"{row['full_name']}:")
            print(f"  - Kills → Underdog line: {row.get('kills', 'N/A')}, Model prediction: {row.get('pred_kills', 'N/A')}")
            print(f"  - Assists → Underdog line: {row.get('assists', 'N/A')}, Model prediction: {row.get('pred_assists', 'N/A')}")
            print(f"  - Deaths → Underdog line: {row.get('deaths', 'N/A')}, Model prediction: {row.get('pred_deaths', 'N/A')}")
            print("-" * 50)

        return results_df

# Example Usage
if __name__ == "__main__":
    MODEL_PATH = "optimized_model.pkl"  # Path to trained model
    UNDERDOG_DATA_PATH = "underdog_props.csv"  # Underdog Fantasy betting lines
    TRAINING_DATA_PATH = "train_features.csv"  # Full Oracle's Elixir dataset

    predictor = UnderdogPredictor(MODEL_PATH, UNDERDOG_DATA_PATH, TRAINING_DATA_PATH)
    
    # Run predictions
    preds = predictor.predict()
    
    # Compare predictions with Underdog betting lines
    results = predictor.compare_to_underdog_lines(preds)
    
    # Save results to CSV
    results.to_csv("underdog_predictions.csv", index=False)
    print("Predictions vs. Underdog lines saved to underdog_predictions.csv.")
