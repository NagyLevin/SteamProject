
import pandas as pd

df = pd.read_csv( #read csv file
    "data/raw/steam-200k.csv",
    header=None,
    names=["user_id", "game_name", "behavior", "value", "extra"] #gve names to the colums
)

#print(df.head()) #show data content

#print(df.shape) #show dimensions of data

#print(df.info()) #show data stats

#print(df.dtypes) # show data types

#print(df.describe()) #show data statiscitc

#print(df["user_id"].nunique()) #how many unique users are there

#print(df["game_name"].nunique()) #how many games

#print(df["behavior"].value_counts())