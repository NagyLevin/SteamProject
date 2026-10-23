
import pandas as pd
from pathlib import Path

def main():
    # Load original dataset
    df = pd.read_csv(
        "data/raw/steam-200k.csv",
        header=None,
        names=["user_id", "game_name", "behavior", "value", "extra"]
    )

    # Remove extra column
    print("---Remove extra coloum---")
    clean_df = remove_extra_column(df)

    # Verify column removal
    print("---- Extra Column Removal ---")
    print("Original shape:", df.shape)
    print("Cleaned shape:", clean_df.shape)
    print("Remaining columns:", clean_df.columns.tolist())
    print("Extra column removed:", "extra" not in clean_df.columns)

    print("---Remove duplicates---")
    nodupes_clean_df = remove_duplicate_records(clean_df)

    # Verify duplicate remove
    print("--- Duplicate Removal Verify ---")
    print("Duplicates before:", df.duplicated().sum())
    print("Duplicates after:", nodupes_clean_df.duplicated().sum())
    print("Original rows:", len(df))
    print("Cleaned rows:", len(nodupes_clean_df))
    print("Removed rows:", len(df) - len(nodupes_clean_df))
    print("Cleaned shape:", nodupes_clean_df.shape)

    print("--- Remove Repeated Play Records ---")
    maxplay_clean_df = keep_max_playtime(nodupes_clean_df)


    print("Duplicates before:", nodupes_clean_df.duplicated().sum())
    print("Duplicates after:", maxplay_clean_df.duplicated().sum())
    print("Original rows:", len(nodupes_clean_df))
    print("Cleaned rows:", len(maxplay_clean_df))
    print("Removed rows:", len(nodupes_clean_df) - len(maxplay_clean_df))
    print("Cleaned shape:", maxplay_clean_df.shape)

    output_path = Path("data/processed/clean_steamdata.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    #maxplay_clean_df.to_csv(output_path, index=False)
    maxplay_clean_df.to_csv(output_path, index=False, header=False)

def remove_extra_column(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the unnecessary extra column
    return df.drop(columns=["extra"]).copy()


def remove_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    # Remove fully duplicated rows, keeping the first one of them
    return df.drop_duplicates().copy()

def keep_max_playtime(df: pd.DataFrame) -> pd.DataFrame:
    #Select play records
    play_df = df[df["behavior"] == "play"]

    #Find highest playtime index for each user-game pair
    max_indices = play_df.groupby(
        ["user_id", "game_name"]
    )["value"].idxmax()

    #Find redundant play record indices
    indices_to_drop = play_df.index.difference(max_indices)

    #Remove redundant play records
    return df.drop(index=indices_to_drop).copy()

    
if __name__ == "__main__":
    main()
