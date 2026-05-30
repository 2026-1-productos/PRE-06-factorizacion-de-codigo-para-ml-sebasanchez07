"""Funciones utilitarias para cargar y preparar datos."""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_wine_data(test_size=0.25, random_state=123456):
    """Carga el dataset de vino y divide en train/test."""

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    return train_test_split(x, y, test_size=test_size, random_state=random_state)
