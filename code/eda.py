
import pandas as pd




def main():
    # Load user-level dataset
    df = pd.read_csv("data/processed/user_features.csv")

    # Display dataset information
    print("--- User-Level Dataset Information ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Calculate feature statistics
    statistics = get_feature_statistics(df)

    # Display descriptive statistics
    print("--- Descriptive Statistics ---")
    print(statistics.to_string())


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    # Select numerical behavioral features
    feature_df = df.drop(columns=["user_id"])

    # Calculate descriptive statistics
    return feature_df.describe().T #Transpon

if __name__ == "__main__":
    main()
