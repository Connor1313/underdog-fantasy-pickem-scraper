import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class FeatureEngineering:
    def __init__(self, df):
        self.df = df.copy()
        self.label_encoders = {}

    def select_features(self):
        """Select relevant features for training."""
        features = [
            "league", "year", "split", "patch", "side", "position",
            "golddiffat25", "xpdiffat25", "csdiffat25",
            "kills", "assists", "deaths",  # Past performance data
            "opp_killsat25", "opp_assistsat25", "opp_deathsat25"
        ]
        target = ["killsat25", "assistsat25", "deathsat25"]

        return self.df[features].copy(), self.df[target].copy()

    def encode_categorical(self, df):
        """Encode categorical variables into numerical format."""
        df = df.copy()
        for col in ["league", "split", "side", "position"]:
            le = LabelEncoder()
            df.loc[:, col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def split_data(self, X, Y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        return X_train, X_test, Y_train, Y_test

# Example Usage
if __name__ == "__main__":
    df = pd.read_csv("cleaned_oracle_elixir.csv")
    fe = FeatureEngineering(df)
    X, Y = fe.select_features()
    X_encoded = fe.encode_categorical(X)

    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = fe.split_data(X_encoded, Y)

    # Save datasets
    X_train.to_csv("train_features.csv", index=False)
    X_test.to_csv("test_features.csv", index=False)
    Y_train.to_csv("train_targets.csv", index=False)
    Y_test.to_csv("test_targets.csv", index=False)

    print("Training and testing datasets saved.")
