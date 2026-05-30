#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

import os
import sys

# Agregar la carpeta raiz al path para poder importar src
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from sklearn.neighbors import KNeighborsRegressor

from src.data_utils import load_wine_data
from src.trainer import save_model, train_and_evaluate

# Cargar datos
x_train, x_test, y_train, y_test = load_wine_data()

# Lista de hiperparametros a probar
param_grid = [3, 5, 7, 9, 11]

best_model = None
best_r2 = -float("inf")

for n_neighbors in param_grid:
    print(f"\nProbando: n_neighbors={n_neighbors}")

    estimator = KNeighborsRegressor(n_neighbors=n_neighbors)
    estimator = train_and_evaluate(estimator, x_train, x_test, y_train, y_test)

    # Evaluar en test para comparar
    r2_test = estimator.score(x_test, y_test)

    if r2_test > best_r2:
        best_r2 = r2_test
        best_model = estimator

# Guardar el mejor modelo
save_model(best_model, "models/estimator.pkl")
