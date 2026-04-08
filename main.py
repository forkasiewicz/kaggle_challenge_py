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

    def min_max(self: "DataProcessor", *columns: str) -> None:
        for column in columns:
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            self.df[column] = (self.df[column] - col_min) / (col_max - col_min)

    def standardize(self: "DataProcessor", *columns: str) -> None:
        for column in columns:
            col_mean = self.df[column].mean()
            stdev = self.df[column].std()
            self.df[column] = (self.df[column] - col_mean) / stdev

    def split_data(self: "DataProcessor", train_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        randomized_df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        row_count: int = len(randomized_df)
        train_row_count: int = int(train_size * row_count)

        hidden_df = randomized_df.iloc[train_row_count:].copy()
        train_df = randomized_df.iloc[:train_row_count].copy()
        return train_df, hidden_df

class LinearRegression:
    def __init__(self: "LinearRegression", df: pd.DataFrame, target: str) -> None:
        self.X: np.ndarray = df.drop(columns=target).to_numpy(dtype=np.float32)
        self.y: np.ndarray = df[target].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.w: np.ndarray = np.zeros((self.X.shape[1], 1), dtype=np.float32)
        self.b: float = 0.0

    def predict(self: "LinearRegression", X: np.ndarray) -> np.ndarray:
        return (X @ self.w) + self.b

    def calculate_rmse(self: "LinearRegression", X: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.predict(X)

        return np.sqrt(np.mean((y_hat - y) ** 2))

    def calculate_r2(self: "LinearRegression", X: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot

    def train(self: "LinearRegression", iterations: int = 1000, learning_step: float = 0.0001) -> None:
        m: int = self.X.shape[0]

        for _ in range(iterations):
            y_hat = self.predict(self.X)
            error = y_hat - self.y

            dw: np.ndarray = (1 / m) * (self.X.T @ error)
            db = float(np.mean(error))

            self.w = self.w - (dw * learning_step)
            self.b -= db * learning_step

if __name__ == "__main__":
    csv: DataProcessor = DataProcessor(file_path="diamonds.csv")
    csv.one_hot("cut", "color", "clarity")
    # csv.min_max("carat", "depth", "table", "x", "y", "z")
    csv.standardize("carat", "depth", "table", "x", "y", "z")

    train_df, hidden_df = csv.split_data(train_size=0.80)
    target = "price"

    model = LinearRegression(train_df, target=target)
    model.train(iterations=20000, learning_step=0.01)

    hidden_X: np.ndarray = hidden_df.drop(columns=target).to_numpy(dtype=np.float32)
    hidden_y: np.ndarray = hidden_df[target].to_numpy(dtype=np.float32).reshape(-1, 1)

    print(f"RMSE: {model.calculate_rmse(hidden_X, hidden_y)}")
    print(f"  R2: {model.calculate_r2(hidden_X, hidden_y)}")

    # to_csv("answers_Forkasiewicz.csv")
