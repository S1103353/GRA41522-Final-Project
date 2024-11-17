from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import statsmodels.api as sm
import requests
from io import StringIO

# Base Class: DataLoader
class DataLoader(ABC):
    def __init__(self):
        self._X = None
        self._y = None

    @abstractmethod
    def load_data(self):
        pass

    def add_constant(self):
        if self._X is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")
        self._X = np.hstack((np.ones((self._X.shape[0], 1)), self._X))

    @property
    def X(self):
        if self._X is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")
        return self._X

    @property
    def y(self):
        if self._y is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")
        return self._y

    @property
    def X_transpose(self):
        return self.X.T

# Subclass: StatsmodelsLoader
class StatsmodelsLoader(DataLoader):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name

    def load_data(self):
        try:
            data = sm.datasets.get_rdataset(self.dataset_name).data
            self._X = data.iloc[:, :-1].to_numpy()
            self._y = data.iloc[:, -1].to_numpy()
            print(f"Loaded statsmodels dataset: {self.dataset_name}")
        except ValueError:
            print(f"Dataset '{self.dataset_name}' not found in statsmodels. Please verify the dataset name.")

# Subclass: CSVLoader
class CSVLoader(DataLoader):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def load_data(self):
        try:
            data = pd.read_csv(self.file_path)
            self._X = data.iloc[:, :-1].to_numpy()
            self._y = data.iloc[:, -1].to_numpy()
            print(f"Loaded local CSV file: {self.file_path}")
        except FileNotFoundError:
            print(f"File '{self.file_path}' not found. Please verify the file path.")

# Subclass: OnlineCSVLoader
class OnlineCSVLoader(DataLoader):
    def __init__(self, url):
        super().__init__()
        self.url = url

    def load_data(self):
        try:
            response = requests.get(self.url)
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch data from {self.url}")
            csv_data = StringIO(response.text)
            data = pd.read_csv(csv_data)
            self._X = data.iloc[:, :-1].to_numpy()
            self._y = data.iloc[:, -1].to_numpy()
            print(f"Loaded online CSV file: {self.url}")
        except Exception as e:
            print(f"Error loading online data: {e}")
