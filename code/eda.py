
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



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


    print("--- Games Purchased Frequency ---")
    print(
        df["games_purchased"]
        .value_counts()
        .sort_index()
        .head(15)
    )

    # Check high percentiles
    print("--- Games Purchased Percentiles ---")
    print(
        df["games_purchased"].quantile([0.90, 0.95, 0.99]) #quantile counting
    )
    #plot histogram oin game distribution
    plot_games_purchased(df)




def plot_games_purchased(df: pd.DataFrame) -> None:
    # Define game ownership categories
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, float("inf")]

    labels = [
        "1", "2", "3", "4-5", "6-10",
        "11-20", "21-50", "51-100", "101+"
    ]

    # Categorize users by purchased games
    game_groups = pd.cut(
        df["games_purchased"],
        bins=bins,
        labels=labels
    )

    # Count users in each category
    counts = game_groups.value_counts().sort_index()

    # Print category frequencies
    print("--- Games Purchased Categories ---")

    for category, count in counts.items():
        percentage = count / len(df) * 100
        print(f"{category}: {count} users ({percentage:.1f}%)")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(counts.index.astype(str), counts.values)

    # Display user counts and percentages
    ax.bar_label(
        bars,
        labels=[
            f"{count} ({count / len(df) * 100:.1f}%)"
            for count in counts.values
        ],
        padding=4
    )

    ax.set_xlim(0, counts.max() * 1.3)
    ax.invert_yaxis()

    # Add chart labels
    ax.set_title("Distribution of Games Purchased")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Games Purchased")

    # Save chart
    plt.tight_layout()
    plt.savefig("figures/games_purchased_groups.png", dpi=150)
    plt.close()




def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    # Select numerical behavioral features
    feature_df = df.drop(columns=["user_id"])

    # Calculate descriptive statistics
    return feature_df.describe().T #Transpon

if __name__ == "__main__":
    main()
