#
# Busque los mejores parametros de un modelo ElasticNet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Consideere los siguentes valores de los hiperparametros y obtenga el
# mejor modelo.
# (alpha, l1_ratio):
#    (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

import os
import sys

# Agregar la carpeta raiz al path para poder importar src
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from sklearn.linear_model import ElasticNet

from src.data_utils import load_wine_data
from src.trainer import save_model, train_and_evaluate

# Cargar datos
x_train, x_test, y_train, y_test = load_wine_data()

# Lista de hiperparametros a probar
param_grid = [
    (0.5, 0.5),
    (0.2, 0.2),
    (0.1, 0.1),
    (0.1, 0.05),
    (0.3, 0.2),
]

best_model = None
best_r2 = -float("inf")

for alpha, l1_ratio in param_grid:
    print(f"\nProbando: alpha={alpha}, l1_ratio={l1_ratio}")

    estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)
    estimator = train_and_evaluate(estimator, x_train, x_test, y_train, y_test)

    # Evaluar en test para comparar
    r2_test = estimator.score(x_test, y_test)

    if r2_test > best_r2:
        best_r2 = r2_test
        best_model = estimator

# Guardar el mejor modelo
save_model(best_model, "models/estimator.pkl")
