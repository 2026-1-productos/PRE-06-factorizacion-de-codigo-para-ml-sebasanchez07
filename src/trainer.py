"""Funciones para entrenar y guardar modelos."""

import os

import joblib


def train_and_evaluate(estimator, x_train, x_test, y_train, y_test):
    """Entrena un modelo y evalua en train y test."""
    from .metrics import compute_metrics, print_metrics

    # Entrenar
    estimator.fit(x_train, y_train)

    print()
    print(estimator, ":", sep="")

    # Metricas de entrenamiento
    y_pred_train = estimator.predict(x_train)
    train_metrics = compute_metrics(y_train, y_pred_train)
    print_metrics(train_metrics, title="Metricas de entrenamiento")

    # Metricas de testing
    y_pred_test = estimator.predict(x_test)
    test_metrics = compute_metrics(y_test, y_pred_test)
    print_metrics(test_metrics, title="Metricas de testing")

    return estimator


def save_model(model, output_path="models/estimator.pkl"):
    """Guarda un modelo entrenado en disco."""

    # Crear la carpeta models/ si no existe
    folder = os.path.dirname(output_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Guardar el modelo
    joblib.dump(model, output_path)
    print()
    print(f"Modelo guardado en: {output_path}")
