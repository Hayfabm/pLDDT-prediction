"""utils file"""
from typing import List, Tuple

import pandas as pd


# data generation
def create_dataset(data_path: str) -> Tuple[List[str], List[int]]:
    dataset = pd.read_csv(data_path)
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # shuffle the dataset
    return list(dataset["sequences"]), list(dataset["labels"])
