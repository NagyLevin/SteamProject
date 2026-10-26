
import pandas as pd

def main():
    df = pd.read_csv( #read csv file
        "data/raw/steam-200k.csv",
        header=None,
        names=["user_id", "game_name", "behavior", "value", "extra"] #gve names to the colums
    )

    print(df.head(50)) #show data content

    print(df.shape) #show dimensions of data

    print(df.info()) #show data stats

    print(df.dtypes) # show data types

    print(df.describe()) #show data statiscitc

    print(df["user_id"].nunique()) #how many unique users are there

    print(df["game_name"].nunique()) #how many games

    print(df["behavior"].value_counts())

    first_user = df["user_id"].iloc[0] #firstuserdata
    get_users_games(df, first_user, 12)
    
    get_user_statistics(df,first_user)
   
def get_user_statistics(
        df: pd.DataFrame,
        user_id: int
    ) -> None:
        user_records = df[df["user_id"] == user_id]

        if user_records.empty:
            print(f"User {user_id} not found.")
            return

        print(f"User Statistics for id: {user_id}")
        print(f"Total records: {len(user_records)}")
        print(f"Unique games: {user_records['game_name'].nunique()}")

def get_users_games(
        df: pd.DataFrame,
        user_id: int,
        record_head: int = 12
    ) -> None:
        user_records = df[df["user_id"] == user_id]

        if user_records.empty:
            print(f"User {user_id} not found.")
            return

        print(
            user_records[["user_id", "game_name", "behavior", "value"]]
            .head(record_head)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()







