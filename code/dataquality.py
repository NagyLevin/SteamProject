import pandas as pd

def main():
    df = pd.read_csv( #read csv file
        "data/raw/steam-200k.csv",
        header=None,
        names=["user_id", "game_name", "behavior", "value", "extra"] #gve names to the colums
    )

    missing_summary = get_missing_values(df)
    #print(missing_summary)
    
    #ceck for duplicates
    duplicate_dict = get_duplicates(df)

    for metric, value in duplicate_dict.items():
        print(f"{metric}: {value}")

    examples, behavior_counts = inspect_duplicate_records(df, duplicate_dict["Duplicate rows"])

    print("Duplicates by behavior:")
    print(behavior_counts)

    print("Example duplicate records:")
    #print(examples.to_string(index=False))

    #see if we have same game twice with different play time
    print(f"behaviour count:{behavior_counts}")

    play_records = df[df["behavior"] == "play"] #get play

    repeated_play_records = play_records[
        play_records.duplicated(
            subset=["user_id", "game_name"], #only compare these
            keep=False #only keep the duplicates
        )
    ]

    print("Repeated User-Game Play Records")
    print("Matching rows:", len(repeated_play_records))
    print(repeated_play_records.head(10).to_string(index=False))


def inspect_duplicate_records(
    df: pd.DataFrame,
    sample_size: int = 12
) -> tuple[pd.DataFrame, pd.Series]:

    # Select all occurrences of duplicated records and keep them
    duplicate_rows = df[df.duplicated(keep=False)]

    # Sort to display matching records next to each other
    duplicate_rows = duplicate_rows.sort_values(
        by=["user_id", "game_name", "behavior", "value"] #like in the csv file
    )

    # Count redundant rows by behavior type
    duplicates_by_behavior = (
        df[df.duplicated()]["behavior"].value_counts()
    )

    return duplicate_rows.head(sample_size), duplicates_by_behavior #duplicate text and number

def get_duplicates(df: pd.DataFrame) -> dict:
    duplicate_count = int(df.duplicated().sum()) #checks if the row was there before, and counts the duplicated rows
    total_rows = len(df)

    duplicate_percentage = (
        duplicate_count / total_rows * 100
        if total_rows > 0 else 0.0
    )

    return {
        "Total rows": total_rows,
        "Duplicate rows": duplicate_count,
        "Duplicate (%)": round(duplicate_percentage, 2)
    }

def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_percent = df.isna().mean() * 100

    return pd.DataFrame({
        "Missing Count": missing_count,
        "Missing (%)": missing_percent.round(2)
    })



if __name__ == "__main__":
    main()

