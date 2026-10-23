import pandas as pd
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv( #read csv file
        #"data/raw/steam-200k.csv",
        "data/processed/clean_steamdata.csv",
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
    
    value_checked_dict = value_check(df)

    # Value Quality Assessment

    for metric, value in value_checked_dict.items():
        print(f"{metric}: {value}")

    
    # Check purchase-play relationships
    relationship_summary = get_purchase_play_relationships(df)

    

    for metric, value in relationship_summary.items():
        print(f"{metric}: {value}")

     # Validate the extra column
    extra_summary = find_extra_column(df)

    for metric, value in extra_summary.items():
        print(f"{metric}: {value}")



def get_purchase_play_relationships(df: pd.DataFrame) -> dict:

        print("--- Purchase-Play Relationships ---")

        # Create sets of unique user-game pairs
        purchase_pairs = set(
            df[df["behavior"] == "purchase"]
            [["user_id", "game_name"]]
            .itertuples(index=False, name=None)
        )

        play_pairs = set(
            df[df["behavior"] == "play"]
            [["user_id", "game_name"]]
            .itertuples(index=False, name=None)
        )

        # Compare the two sets
        both = purchase_pairs & play_pairs #both
        purchase_only = purchase_pairs - play_pairs #only buy
        play_only = play_pairs - purchase_pairs #only play

        return {
            "Unique user-game pairs": len(purchase_pairs | play_pairs),
            "Purchase and play": len(both),
            "Purchase only": len(purchase_only),
            "Play only": len(play_only)
        }

def find_extra_column(df: pd.DataFrame) -> dict:
    extra_col = df["extra"] #last coloum called extra
    print("--- Extra Column Validation ---")

    return {
        "Unique values": extra_col.nunique(dropna=False), #finds unique values and missing values
        "Value distribution": extra_col.value_counts(dropna=False).to_dict(), 
        "Missing values": int(extra_col.isna().sum()),
        "All values are zero": bool(extra_col.eq(0).all())
    }

def value_check(df: pd.DataFrame) -> dict:
    print("---Value Quality Assessment---")
    # Separate purchase and play records
    purchase_df = df[df["behavior"] == "purchase"]
    play_df = df[df["behavior"] == "play"]

    # Check for unexpected behavior types behaivour shoud be purchase or play nothing else
    invalid_behavior = ~df["behavior"].isin(["purchase", "play"]) 

    # Check invalid numeric values #records that are inf are not allowed
    invalid_numeric = ~np.isfinite(df["value"]) #~ flip the logical value 

    # Purchase records should have value = 1.0
    invalid_purchase = purchase_df["value"] != 1.0

    # Check playtime values playtime cant be negative or 0
    negative_playtime = play_df["value"] < 0
    zero_playtime = play_df["value"] == 0

    return { #return more things at the same time 
        "Invalid behavior records": int(invalid_behavior.sum()),
        "Invalid numeric values": int(invalid_numeric.sum()),
        "Invalid purchase values": int(invalid_purchase.sum()),
        "Negative playtime records": int(negative_playtime.sum()),
        "Zero playtime records": int(zero_playtime.sum()),
        "Minimum playtime": play_df["value"].min(),
        "Maximum playtime": play_df["value"].max()
    }



def inspect_duplicate_records(
    df: pd.DataFrame,
    sample_size: int = 12
) -> tuple[pd.DataFrame, pd.Series]:
    print("Inspect duplicate records")
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
    print("---Get missing values---")
    missing_count = df.isna().sum()
    missing_percent = df.isna().mean() * 100

    return pd.DataFrame({
        "Missing Count": missing_count,
        "Missing (%)": missing_percent.round(2)
    })



if __name__ == "__main__":
    main()

