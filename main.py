""" 
- main.py
- Date: 8/4/2026
- Author: @forkasiewicz
"""

import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self: "DataProcessor", file_path: str) -> None:
        self.file_path: str = file_path
        self.df: pd.DataFrame = pd.read_csv(file_path, index_col=0)

    def one_hot(self: "DataProcessor", *columns: str) -> None:
        for column in columns:
            one_hot_columns: pd.DataFrame = pd.get_dummies(self.df[column], prefix=column)
            self.df = self.df.drop(columns=[column])
            self.df = self.df.join(one_hot_columns)

    def scale(self: "DataProcessor", *columns: str) -> None:
        for column in columns:
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            self.df[column] = (self.df[column] - col_min) / (col_max - col_min)

    def split_data(self: "DataProcessor", train_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        row_count: int = len(self.df)
        train_row_count: int = int(train_size * row_count)

        hidden_df: pd.DataFrame = self.df.iloc[train_row_count:].copy()
        train_df: pd.DataFrame = self.df.iloc[:train_row_count].copy()
        return train_df, hidden_df

class LinearRegression:
    def __init__(self: "LinearRegression", df: pd.DataFrame, target: str) -> None:
        self.X: np.ndarray = df.drop(columns=target).to_numpy()
        self.y: np.ndarray = df[target].to_numpy()
        self.w: np.ndarray = np.zeros((self.X.shape[1], 1))
        self.b: float = 0.0

    def predict(self: "LinearRegression", X: np.ndarray) -> np.ndarray:
        return (X @ self.w) + self.b
        
    def calculate_rmse(self: "LinearRegression", X: np.ndarray, y: np.ndarray) -> float:
        y_hat: np.ndarray = self.predict(X)
        mse = np.mean((y_hat - y) ** 2)
        return np.sqrt(mse)

if __name__ == "__main__":
    csv: DataProcessor = DataProcessor("diamonds.csv")
    csv.one_hot("cut", "color", "clarity")
    csv.scale("carat", "depth", "table", "x", "y", "z")
    
    train_df, hidden_df = csv.split_data(train_size=0.8)

    # linear_regression = LinearRegression(train_df, target="price")

    # to_csv("answers_Forkasiewicz.csv")

"""
initialize weights and bias (all zeros)

loop:
    calculate y_hat (matrix multiplication) y_hat = X@w + b
    calculate rmse
    calculate derivatives of w and b 
    subtract gradient * learning_rate from w and b

for checking the results I would want a separate method that would use the
weights for a final rerun using the hidden dataset and compare the prediction
against the target. Essentially reinitialize our trained linearregression with
a different set.
"""
