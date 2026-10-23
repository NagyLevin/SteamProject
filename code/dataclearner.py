
import pandas as pd


def main():
    # Load original dataset
    df = pd.read_csv(
        "data/raw/steam-200k.csv",
        header=None,
        names=["user_id", "game_name", "behavior", "value", "extra"]
    )

    # Remove extra column
    clean_df = remove_extra_column(df)

    # Verify column removal
    print("---- Extra Column Removal ---")
    print("Original shape:", df.shape)
    print("Cleaned shape:", clean_df.shape)
    print("Remaining columns:", clean_df.columns.tolist())
    print("Extra column removed:", "extra" not in clean_df.columns)



def remove_extra_column(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the unnecessary extra column
    return df.drop(columns=["extra"]).copy()

    
if __name__ == "__main__":
    main()
