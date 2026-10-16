
import pandas as pd

df = pd.read_csv( #read csv file
    "data/raw/steam-200k.csv",
    header=None,
    names=["user_id", "game_name", "behavior", "value", "extra"] #gve names to the colums
)

print(df.head()) #show data content
