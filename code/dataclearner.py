
import pandas as pd


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
    print("Duplicates after:", nodupes_clean_df)
    print("Original rows:", len(df))
    print("Cleaned rows:", len(nodupes_clean_df))
    print("Removed rows:", len(df) - len(nodupes_clean_df))
    print("Cleaned shape:", nodupes_clean_df.shape)


def remove_extra_column(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the unnecessary extra column
    return df.drop(columns=["extra"]).copy()


def remove_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    # Remove fully duplicated rows, keeping the first one of them
    return df.drop_duplicates().copy()
    
if __name__ == "__main__":
    main()
