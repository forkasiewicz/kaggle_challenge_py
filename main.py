""" 
- main.py
- Date: 8/4/2026
- Author: @forkasiewicz
"""

import pandas as pd

class Data:
    def __init__(self: Data, file_path: str):
        self.file_path: str = file_path
        self.df: pd.DataFrame = pd.read_csv(file_path, index_col=0)

    def one_hot(self: Data, column: str) -> Data:
        one_hot_columns: pd.DataFrame = pd.get_dummies(self.df[column], prefix=column)
        self.df = self.df.drop(columns=[column])
        self.df = self.df.join(one_hot_columns)
        return self

    def scale(self: Data):
        pass

    def split_data(self: Data, train_size: float):
        pass

class LinearRegression:
    def __init__(self: LinearRegression, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.X: pd.DataFrame = X
        self.y: pd.DataFrame = y


if __name__ == "__main__":
    data: Data = Data("diamonds.csv")
    data = data.one_hot("cut").one_hot("color").one_hot("clarity")

    # data.df.to_csv("answers_kacper_forkasiewicz.csv")
    y = data.df["price"]
    X = data.df.drop(columns="price")
    linear_regression: LinearRegression = LinearRegression(X, y)

"""
weight (w) - the multiplier (e.g. how much the price depends on this)
bias (b) - base price

example:
price = (w * height) + b

price = (w * height) + (w * width) + b
"""

"""
A data frame with 53940 rows and 10 variables:
price - price in US dollars ($326-$18,823)
carat - weight of the diamond (0.2-5.01)
cut - quality of the cut (Fair, Good, Very Good, Premium, Ideal)
color - diamond colour, from J (worst) to D (best)
clarity - a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
x - length in mm (0-10.74)
y - width in mm (0-58.9)
z - depth in mm (0-31.8)
depth - total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43-79)
table - width of top of diamond relative to widest point (43-95)
"""

"""
"","carat","cut","color","clarity","depth","table","price","x","y","z"
"1",0.23,"Ideal","E","SI2",61.5,55,326,3.95,3.98,2.43
"2",0.21,"Premium","E","SI1",59.8,61,326,3.89,3.84,2.31
"3",0.23,"Good","E","VS1",56.9,65,327,4.05,4.07,2.31
"4",0.29,"Premium","I","VS2",62.4,58,334,4.2,4.23,2.63
"5",0.31,"Good","J","SI2",63.3,58,335,4.34,4.35,2.75
"6",0.24,"Very Good","J","VVS2",62.8,57,336,3.94,3.96,2.48
"7",0.24,"Very Good","I","VVS1",62.3,57,336,3.95,3.98,2.47
"8",0.26,"Very Good","H","SI1",61.9,55,337,4.07,4.11,2.53
"9",0.22,"Fair","E","VS2",65.1,61,337,3.87,3.78,2.49
"10",0.23,"Very Good","H","VS1",59.4,61,338,4,4.05,2.39
"11",0.3,"Good","J","SI1",64,55,339,4.25,4.28,2.73
"12",0.23,"Ideal","J","VS1",62.8,56,340,3.93,3.9,2.46
"""
