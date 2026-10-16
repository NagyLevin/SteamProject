import pandas as pd

def main():
    df = pd.read_csv( #read csv file
        "data/raw/steam-200k.csv",
        header=None,
        names=["user_id", "game_name", "behavior", "value", "extra"] #gve names to the colums
    )
    missing_summary = get_missing_values(df)
    print(missing_summary)
    #TODO continue with duplicate check


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_percent = df.isna().mean() * 100

    return pd.DataFrame({
        "Missing Count": missing_count,
        "Missing (%)": missing_percent.round(2)
    })



if __name__ == "__main__":
    main()

