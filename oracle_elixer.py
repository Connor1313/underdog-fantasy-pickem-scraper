import pandas as pd
from sklearn.preprocessing import LabelEncoder

class FeatureEngineering:
    def __init__(self, df):
        self.df = df.copy()  # Make a full copy to avoid warnings
        self.label_encoders = {}

    def select_features(self):
        """Select relevant features for training."""
        features = ["league", "year", "split", "patch", "side", "position",
                    "golddiffat25", "xpdiffat25", "csdiffat25"]
        target = ["killsat25", "assistsat25", "deathsat25"]
        return self.df[features].copy(), self.df[target].copy()  # Return copies to prevent warnings

    def encode_categorical(self, df):
        """Encode categorical variables into numerical format."""
        df = df.copy()  # Ensure we modify a copy

        for col in ["league", "split", "side", "position"]:
            le = LabelEncoder()
            df.loc[:, col] = le.fit_transform(df[col])  # Explicitly use .loc[]
            self.label_encoders[col] = le  # Store encoders for future use

        return df

# Example Usage
if __name__ == "__main__":
    df = pd.read_csv("cleaned_oracle_elixir.csv")
    fe = FeatureEngineering(df)
    X, Y = fe.select_features()
    X_encoded = fe.encode_categorical(X)

    X_encoded.to_csv("processed_features.csv", index=False)
    Y.to_csv("targets.csv", index=False)
    print("Processed features and targets saved.")
