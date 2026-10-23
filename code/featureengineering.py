
import pandas as pd


def main():
    # Load cleaned dataset
    df = pd.read_csv(
        "data/processed/clean_steamdata.csv",
        header=None,
        names=["user_id", "game_name", "behavior", "value"]
    )

    # Create user-level dataset
    user_df = create_user_features(df)

    # Display first users for cecking out how to do it
    print("--- User-Level Features ---")
    print(user_df.head(5).to_string(index=False))

    # Verify feature engineering
    print("--- Feature Engineering Verification ---")
    print("Original unique users:", df["user_id"].nunique())
    print("User-level rows:", len(user_df))
    print("Duplicate user IDs:", df["user_id"].nunique()-len(user_df))
    print("Total games purchased:", user_df["games_purchased"].sum())
    print("Total games played:", user_df["games_played"].sum())
    print("User-level shape:", user_df.shape)

   
def create_user_features(df: pd.DataFrame) -> pd.DataFrame:
    # Dictionary to store user features
    user_features = {}

    # Iterate through each user
    for user_id, user_records in df.groupby("user_id"):

        # Separate purchase and play records
        purchase_records = user_records[
            user_records["behavior"] == "purchase"
        ]

        play_records = user_records[
            user_records["behavior"] == "play"
        ]

        # Count unique games
        games_purchased = purchase_records["game_name"].nunique()
        games_played = play_records["game_name"].nunique()

        # Calculate playtime statistics
        total_playtime = play_records["value"].sum()

        avg_playtime = (
            play_records["value"].mean()
            if games_played > 0 else 0.0
        )

        max_playtime = (
            play_records["value"].max()
            if games_played > 0 else 0.0
        )

        # Calculate play ratio
        play_ratio = (
            games_played / games_purchased
            if games_purchased > 0 else 0.0
        )

        # Store features for the current user
        user_features[user_id] = {
            "games_purchased": games_purchased,
            "games_played": games_played,
            "total_playtime": total_playtime,
            "avg_playtime": avg_playtime,
            "max_playtime": max_playtime,
            "play_ratio": play_ratio
        }

    # Convert dictionary to DataFrame
    user_df = pd.DataFrame.from_dict(
        user_features,
        orient="index"
    )

    user_df.index.name = "user_id"
    user_df = user_df.reset_index()

    return user_df



if __name__ == "__main__":
    main()
