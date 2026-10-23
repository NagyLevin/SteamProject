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

