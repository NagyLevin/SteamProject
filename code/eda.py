
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
    plot_total_playtime(df)
    comparison = compare_ownership_and_playtime(df)
    print(comparison.round(2).to_string())

      # Calculate feature correlations
    correlation_matrix = get_feature_correlations(df)

    # Display correlation matrix
    print("--- Spearman Correlation Matrix ---")
    print(correlation_matrix.round(2).to_string())
    plot_correlation_heatmap(correlation_matrix)

    print("--- Outlier Analysis ---")
    outlier_results = analyze_outliers(df)
    print(outlier_results.round(2).to_string(index=False))


def analyze_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Select features for outlier analysis
    features = ["games_purchased", "total_playtime"] #tow coloums that we want to use can be expanded


    # Store analysis results
    results = []

    for feature in features:
        # Calculate quartiles
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)

        # Calculate IQR and boundaries
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Select potential outliers
        outliers = df[
            (df[feature] < lower_bound) |
            (df[feature] > upper_bound)
        ]

        # Calculate outlier count and percentage
        outlier_count = len(outliers)
        outlier_percentage = outlier_count / len(df) * 100

        # Store results
        results.append({
            "feature": feature,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage
        })

    # Convert results to DataFrame
    return pd.DataFrame(results)


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    # Create heatmap
    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    # Add feature names
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))

    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation_matrix.index)

    # Display correlation values
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[i, j]
            text_color = "white" if abs(value) >= 0.7 else "black"

            ax.text(
                j, i, f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color
            )

    # Add color legend
    fig.colorbar(image, ax=ax, label="Spearman Correlation")

    # Add chart title
    ax.set_title("Spearman Correlation Heatmap")

    # Save heatmap
    plt.tight_layout()
    plt.savefig("figures/spearman_correlation_heatmap.png", dpi=150)
    plt.close()


def get_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    # Select numerical behavioral features
    feature_df = df.drop(columns=["user_id"])

    # Calculate Spearman correlation matrix
    correlation_matrix = feature_df.corr(method="spearman")

    return correlation_matrix


def compare_ownership_and_playtime(df: pd.DataFrame) -> pd.DataFrame:
    # Define game ownership categories
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, float("inf")]

    labels = [
        "1", "2", "3", "4-5", "6-10",
        "11-20", "21-50", "51-100", "101+"
    ]

    # Categorize users by purchased games
    ownership_groups = pd.cut(
        df["games_purchased"],
        bins=bins,
        labels=labels
    )

    # Calculate playtime statistics for each category
    summary = df.groupby(
        ownership_groups,
        observed=True
    )["total_playtime"].agg(
        ["count", "mean", "median"]
    )

    return summary


def plot_total_playtime(df: pd.DataFrame) -> None:
    # Define playtime categories
    bins = [-0.001, 0, 1, 10, 100, 500, 1000, float("inf")]

    labels = [
        "0", "0-1", "1-10", "10-100",
        "100-500", "500-1000", ">1000"
    ]

    # Categorize users by total playtime
    playtime_groups = pd.cut(
        df["total_playtime"],
        bins=bins,
        labels=labels
    )

    # Count users in each category
    counts = playtime_groups.value_counts().sort_index()

    # Print category frequencies
    print("--- Total Playtime Categories ---")

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
    ax.set_title("Distribution of Total Playtime")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Total Playtime (Hours)")

    # Save chart
    plt.tight_layout()
    plt.savefig("figures/total_playtime_groups.png", dpi=150)
    plt.close()


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
